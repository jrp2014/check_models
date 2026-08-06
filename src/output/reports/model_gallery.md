# Model Output Gallery

Generated on: 2026-08-06 00:39:55 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability             | Total s   | Gen TPS             | Peak GB   | Gen tok   | Observations                                                                             |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|-----------|---------------------|-----------|-----------|------------------------------------------------------------------------------------------|
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `usable`              | 1.59s     | 126 tok/s           | 5.5       | 63        | none                                                                                     |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | 0.70s     | 520 tok/s           | 1.0       | 169       | none                                                                                     |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable`              | 2.92s     | 133 tok/s           | 5.2       | 281       | none                                                                                     |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `usable`              | 29.61s    | 5.06 tok/s          | 25        | 130       | none                                                                                     |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `usable`              | 1.56s     | 60.9 tok/s          | 9.3       | 20        | none                                                                                     |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | 6.36s     | 31.3 tok/s          | 20        | 107       | none                                                                                     |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `usable`              | 1.92s     | 360 tok/s           | 2.2       | 428       | none                                                                                     |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `usable`              | 7.89s     | 51.3 tok/s          | 13        | 301       | none                                                                                     |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `usable`              | 4.86s     | 63.6 tok/s          | 10        | 197       | none                                                                                     |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable`              | 3.82s     | 91.8 tok/s          | 7.7       | 205       | none                                                                                     |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable`              | 14.05s    | 52.5 tok/s          | 63        | 172       | none                                                                                     |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `usable`              | 6.53s     | 32.1 tok/s          | 19        | 77        | none                                                                                     |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 5.82s     | 34.1 tok/s          | 18        | 99        | none                                                                                     |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `usable`              | 0.97s     | 330 tok/s           | 3.0       | 103       | none                                                                                     |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | 1.26s     | 186 tok/s           | 4.1       | 108       | none                                                                                     |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | 12.22s    | 16.6 tok/s          | 15        | 155       | none                                                                                     |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable`              | 1.44s     | 285 tok/s           | 3.0       | 112       | none                                                                                     |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 4.26s     | 69.1 tok/s          | 9.8       | 155       | none                                                                                     |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              | 4.68s     | 63.0 tok/s          | 10        | 156       | none                                                                                     |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 2.21s     | 187 tok/s           | 4.5       | 178       | none                                                                                     |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable`              | 4.98s     | 53.5 tok/s          | 11        | 174       | none                                                                                     |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `usable`              | 8.58s     | 30.9 tok/s          | 18        | 199       | none                                                                                     |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `usable`              | 29.69s    | 5.95 tok/s          | 23        | 158       | none                                                                                     |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | 20.28s    | 67.4 tok/s          | 71        | 305       | none                                                                                     |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | 1.58s     | 61.9 tok/s          | 9.3       | 20        | none                                                                                     |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable`              | 0.90s     | 331 tok/s           | 2.5       | 69        | none                                                                                     |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable`              | 3.12s     | 132 tok/s           | 5.3       | 281       | none                                                                                     |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable`              | 4.03s     | 131 tok/s           | 5.3       | 401       | none                                                                                     |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `usable`              | 12.32s    | 33.3 tok/s          | 19        | 315       | none                                                                                     |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `usable`              | 20.65s    | 18.3 tok/s          | 30        | 305       | none                                                                                     |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 5.59s     | 122 tok/s           | 21        | 326       | none                                                                                     |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `usable`              | 6.68s     | 102 tok/s           | 30        | 299       | none                                                                                     |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `usable`              | 18.79s    | 68.8 tok/s          | 71        | 299       | none                                                                                     |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 4.34s     | 100 tok/s           | 7.0       | 265       | none                                                                                     |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | 19.39s    | 18.2 tok/s          | 30        | 283       | none                                                                                     |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `usable`              | 1.58s     | 127 tok/s           | 5.5       | 63        | none                                                                                     |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable`              | 2.04s     | 132 tok/s           | 5.5       | 143       | none                                                                                     |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | 14.50s    | 43.7 tok/s          | 65        | 172       | none                                                                                     |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable`              | 3.05s     | 66.3 tok/s          | 10        | 104       | none                                                                                     |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | 13.23s    | 29.8 tok/s          | 18        | 303       | none                                                                                     |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `usable`              | 21.54s    | 16.6 tok/s          | 32        | 289       | none                                                                                     |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `usable`              | 4.97s     | 48.4 tok/s          | 17        | 106       | none                                                                                     |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 3.98s     | 129 tok/s           | 16        | 148       | none                                                                                     |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `usable`              | 12.54s    | insufficient sample | 63        | 15        | none                                                                                     |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 6.88s     | 27.2 tok/s          | 19        | 98        | none                                                                                     |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `usable`              | 4.90s     | 62.5 tok/s          | 9.7       | 82        | none                                                                                     |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `usable`              | 0.96s     | 378 tok/s           | 1.8       | 123       | none                                                                                     |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `usable`              | 8.08s     | 32.6 tok/s          | 11        | 180       | none                                                                                     |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `usable`              | 33.30s    | 5.41 tok/s          | 26        | 159       | none                                                                                     |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `usable`              | 8.20s     | 19.7 tok/s          | 10        | 126       | none                                                                                     |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | 7.47s     | 40.2 tok/s          | 15        | 199       | none                                                                                     |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `usable`              | 13.60s    | 20.0 tok/s          | 27        | 204       | none                                                                                     |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `usable`              | 1.00s     | 116 tok/s           | 4.2       | 44        | none                                                                                     |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `usable_with_caveats` | 5.22s     | 77.6 tok/s          | 16        | 221       | Conversation-role control tokens remain visible                                          |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | 14.00s    | 32.2 tok/s          | 19        | 354       | Conversation-role control tokens remain visible                                          |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` | 55.58s    | 4.7 tok/s           | 39        | 241       | Conversation-role control tokens remain visible                                          |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | 5.57s     | 53.4 tok/s          | 29        | 88        | Unrecognised model control tokens remain visible                                         |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | 4.95s     | 67.9 tok/s          | 28        | 89        | Unrecognised model control tokens remain visible                                         |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `usable_with_caveats` | 2.27s     | insufficient sample | 4.6       | 2         | Response is unusually short                                                              |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 13.83s    | 44.3 tok/s          | 14        | 500       | Response appears cut off at the token limit                                              |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 15.72s    | 64.9 tok/s          | 60        | 500       | Response appears cut off at the token limit                                              |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `unusable`            | 10.14s    | 68.2 tok/s          | 20        | 500       | Response appears cut off at the token limit; Internal reasoning block appears incomplete |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `unusable`            | 5.86s     | 126 tok/s           | 5.8       | 500       | Response repeats the same text; Response appears cut off at the token limit              |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                                   | `not_evaluated`       | 0.22s     | -                   | -         | -         | none                                                                                     |
<!-- markdownlint-enable MD034 MD049 -->

## Avoid for This Run

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                        | Usability       | Observations                                                                             |
|--------------------------------------------------------------------------------------------------------------|-----------------|------------------------------------------------------------------------------------------|
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable`      | Response appears cut off at the token limit                                              |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable`      | Response appears cut off at the token limit                                              |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                  | `unusable`      | Response appears cut off at the token limit; Internal reasoning block appears incomplete |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                  | `unusable`      | Response repeats the same text; Response appears cut off at the token limit              |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                        | `not_evaluated` | none                                                                                     |
<!-- markdownlint-enable MD034 MD049 -->

## Lowest-memory Usable Models (Including Caveats)

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability             |   Peak GB |   Gen tok |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|-----------|-----------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              |       1   |       169 |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `usable`              |       1.8 |       123 |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `usable`              |       2.2 |       428 |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable`              |       2.5 |        69 |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `usable`              |       3   |       103 |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable`              |       3   |       112 |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              |       4.1 |       108 |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `usable`              |       4.2 |        44 |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              |       4.5 |       178 |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `usable_with_caveats` |       4.6 |         2 |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable`              |       5.2 |       281 |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable`              |       5.3 |       281 |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable`              |       5.3 |       401 |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `usable`              |       5.5 |        63 |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable`              |       5.5 |       143 |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `usable`              |       5.5 |        63 |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              |       7   |       265 |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable`              |       7.7 |       205 |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `usable`              |       9.3 |        20 |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              |       9.3 |        20 |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `usable`              |       9.7 |        82 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              |       9.8 |       155 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              |      10   |       156 |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable`              |      10   |       104 |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `usable`              |      10   |       197 |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `usable`              |      10   |       126 |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable`              |      11   |       174 |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `usable`              |      11   |       180 |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `usable`              |      13   |       301 |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              |      15   |       199 |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              |      15   |       155 |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `usable_with_caveats` |      16   |       221 |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              |      16   |       148 |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `usable`              |      17   |       106 |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              |      18   |       303 |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `usable`              |      18   |       199 |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              |      18   |        99 |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` |      19   |       354 |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `usable`              |      19   |        77 |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `usable`              |      19   |       315 |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              |      19   |        98 |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              |      20   |       107 |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              |      21   |       326 |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `usable`              |      23   |       158 |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `usable`              |      25   |       130 |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `usable`              |      26   |       159 |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `usable`              |      27   |       204 |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` |      28   |        89 |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` |      29   |        88 |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `usable`              |      30   |       305 |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              |      30   |       283 |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `usable`              |      30   |       299 |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `usable`              |      32   |       289 |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` |      39   |       241 |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable`              |      63   |       172 |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `usable`              |      63   |        15 |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              |      65   |       172 |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              |      71   |       305 |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `usable`              |      71   |       299 |
<!-- markdownlint-enable MD034 MD049 -->

## Fastest Usable Models (Including Caveats)

Fastest valid generation: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 520 tok/s

Average valid generation throughput: 96.6 tok/s

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability             | Gen TPS             |   Gen tok |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------------------|-----------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | 520 tok/s           |       169 |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `usable`              | 378 tok/s           |       123 |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `usable`              | 360 tok/s           |       428 |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable`              | 331 tok/s           |        69 |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `usable`              | 330 tok/s           |       103 |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable`              | 285 tok/s           |       112 |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 187 tok/s           |       178 |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | 186 tok/s           |       108 |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable`              | 133 tok/s           |       281 |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable`              | 132 tok/s           |       143 |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable`              | 132 tok/s           |       281 |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable`              | 131 tok/s           |       401 |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 129 tok/s           |       148 |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `usable`              | 127 tok/s           |        63 |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `usable`              | 126 tok/s           |        63 |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 122 tok/s           |       326 |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `usable`              | 116 tok/s           |        44 |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `usable`              | 102 tok/s           |       299 |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 100 tok/s           |       265 |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable`              | 91.8 tok/s          |       205 |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `usable_with_caveats` | 77.6 tok/s          |       221 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 69.1 tok/s          |       155 |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `usable`              | 68.8 tok/s          |       299 |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | 67.9 tok/s          |        89 |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | 67.4 tok/s          |       305 |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable`              | 66.3 tok/s          |       104 |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `usable`              | 63.6 tok/s          |       197 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              | 63.0 tok/s          |       156 |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `usable`              | 62.5 tok/s          |        82 |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | 61.9 tok/s          |        20 |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `usable`              | 60.9 tok/s          |        20 |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable`              | 53.5 tok/s          |       174 |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | 53.4 tok/s          |        88 |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable`              | 52.5 tok/s          |       172 |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `usable`              | 51.3 tok/s          |       301 |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `usable`              | 48.4 tok/s          |       106 |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | 43.7 tok/s          |       172 |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | 40.2 tok/s          |       199 |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 34.1 tok/s          |        99 |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `usable`              | 33.3 tok/s          |       315 |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `usable`              | 32.6 tok/s          |       180 |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | 32.2 tok/s          |       354 |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `usable`              | 32.1 tok/s          |        77 |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | 31.3 tok/s          |       107 |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `usable`              | 30.9 tok/s          |       199 |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | 29.8 tok/s          |       303 |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 27.2 tok/s          |        98 |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `usable`              | 20.0 tok/s          |       204 |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `usable`              | 19.7 tok/s          |       126 |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `usable`              | 18.3 tok/s          |       305 |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | 18.2 tok/s          |       283 |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `usable`              | 16.6 tok/s          |       289 |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | 16.6 tok/s          |       155 |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `usable`              | 5.95 tok/s          |       158 |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `usable`              | 5.41 tok/s          |       159 |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `usable`              | 5.06 tok/s          |       130 |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` | 4.7 tok/s           |       241 |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `usable`              | insufficient sample |        15 |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `usable_with_caveats` | insufficient sample |         2 |
<!-- markdownlint-enable MD034 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.11`
- `mlx`: `0.32.1.dev20260805+2c46b953d`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.26.0`
- *Python Version:* 3.13.13
- *OS:* Darwin 25.6.0
- *macOS Version:* 26.6
- *GPU/Chip:* Apple M5 Max
- *MLX Device:* Apple M5 Max
- *GPU Architecture:* applegpu_g17s
- *RAM:* 128.0 GB
- *Recommended Working Set:* 108 GB
- *Fused Attention:* Available

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Complete Per-model Evidence

Complete generated or crash evidence for every attempted model.

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### HuggingFaceTB/SmolVLM-Instruct

<details>
<summary>Complete evidence: HuggingFaceTB/SmolVLM-Instruct</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.68s
- *Generation time:* 0.91s
- *Total time:* 1.59s
- *Input validation time:* 0.0016
- *Prompt preparation time:* 0.00175
- *First-token latency:* 0.365
- *Cleanup time:* 0.0707
- *Prompt tokens:* 1,194
- *Generation tokens:* 63
- *Total tokens:* 1,257
- *Prompt throughput (raw):* 3,267 tok/s
- *Generation throughput (raw):* 126 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.30
- *Model-load active memory:* 4.49
- *Post-cleanup active memory:* 0.000116
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 72
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
 Two cats are lying on a pink blanket on a couch. The cat on the left is curled up with its head on the cat on the right's back. The cat on the right is stretched out with its head on the cat on the left's back. There are two remote controls on the blanket.
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Two cats are lying on a pink blanket on a couch. The cat on the left is curled up with its head on the cat on the right's back. The cat on the right is stretched out with its head on the cat on the left's back. There are two remote controls on the blanket.
```

</details>

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

<details>
<summary>Complete evidence: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.34s
- *Generation time:* 0.36s
- *Total time:* 0.70s
- *Input validation time:* 0.00178
- *Prompt preparation time:* 0.00549
- *First-token latency:* 0.0215
- *Cleanup time:* 0.0588
- *Prompt tokens:* 78
- *Generation tokens:* 169
- *Total tokens:* 247
- *Prompt throughput (raw):* 3,632 tok/s
- *Generation throughput (raw):* 520 tok/s
- *Peak memory:* 1.0
- *Active memory:* 0.90
- *Cache memory:* 0.01
- *Model-load active memory:* 0.898
- *Post-cleanup active memory:* 0.000247
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 91
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
In this detailed photograph, two cats are lounging on a pink, wrinkled couch. The cat on the left is a tabby with a mix of black, brown, and white fur, lying on its side with its head resting on its paws. Its tail is curled up behind it, and it has a white patch on its chest. The cat on the right is a tabby with a similar color pattern, but with a slightly lighter coat. This cat is also lying on its side, with its head resting on its front paws and its tail curled up behind it. Both cats have their eyes closed, suggesting they are either sleeping or resting peacefully. In the background, two remote controls are visible, one on the left and one on the right, adding a touch of domesticity to the scene.
</pre>

<details>
<summary>Exact raw output</summary>

```text
In this detailed photograph, two cats are lounging on a pink, wrinkled couch. The cat on the left is a tabby with a mix of black, brown, and white fur, lying on its side with its head resting on its paws. Its tail is curled up behind it, and it has a white patch on its chest. The cat on the right is a tabby with a similar color pattern, but with a slightly lighter coat. This cat is also lying on its side, with its head resting on its front paws and its tail curled up behind it. Both cats have their eyes closed, suggesting they are either sleeping or resting peacefully. In the background, two remote controls are visible, one on the left and one on the right, adding a touch of domesticity to the scene.
```

</details>

</details>

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### Qwen/Qwen3-VL-2B-Instruct

<details>
<summary>Complete evidence: Qwen/Qwen3-VL-2B-Instruct</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.68s
- *Generation time:* 2.22s
- *Total time:* 2.92s
- *Input validation time:* 0.00149
- *Prompt preparation time:* 0.00971
- *First-token latency:* 0.0828
- *Cleanup time:* 0.0742
- *Prompt tokens:* 313
- *Generation tokens:* 281
- *Total tokens:* 594
- *Prompt throughput (raw):* 3,778 tok/s
- *Generation throughput (raw):* 133 tok/s
- *Peak memory:* 5.2
- *Active memory:* 4.3
- *Cache memory:* 0.10
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000558
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 112
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
This is a photograph of two cats resting on a bright pink couch. The scene is calm and cozy, with the cats appearing to be asleep or very relaxed.

- **The Cats**: There are two cats, both with tabby patterns.
    - The cat on the left is a smaller, younger-looking tabby with a mix of black, brown, and grey stripes. It is lying on its back with its paws stretched out, and its eyes are closed.
    - The cat on the right is larger and has a more prominent brown and black striped pattern. It is lying on its side, also with its eyes closed, and appears to be sleeping peacefully.

- **The Setting**: The cats are on a vibrant pink couch, which is the main background of the image. The couch is covered with a soft-looking fabric.

- **The Remotes**: Two remote controls are visible on the couch, placed near the cats.
    - One is a light blue or grey remote control, and the other is a white remote control with a blue button.
    - The remotes are likely for a television or a similar electronic device.

- **Overall Impression**: The image captures a peaceful moment of two cats resting together on a colorful couch. The bright pink color of the couch contrasts with the cats' fur, making them stand out. The overall mood is one of tranquility and comfort.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This is a photograph of two cats resting on a bright pink couch. The scene is calm and cozy, with the cats appearing to be asleep or very relaxed.

- **The Cats**: There are two cats, both with tabby patterns.
    - The cat on the left is a smaller, younger-looking tabby with a mix of black, brown, and grey stripes. It is lying on its back with its paws stretched out, and its eyes are closed.
    - The cat on the right is larger and has a more prominent brown and black striped pattern. It is lying on its side, also with its eyes closed, and appears to be sleeping peacefully.

- **The Setting**: The cats are on a vibrant pink couch, which is the main background of the image. The couch is covered with a soft-looking fabric.

- **The Remotes**: Two remote controls are visible on the couch, placed near the cats.
    - One is a light blue or grey remote control, and the other is a white remote control with a blue button.
    - The remotes are likely for a television or a similar electronic device.

- **Overall Impression**: The image captures a peaceful moment of two cats resting together on a colorful couch. The bright pink color of the couch contrasts with the cats' fur, making them stand out. The overall mood is one of tranquility and comfort.
```

</details>

</details>

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### meta-llama/Llama-3.2-11B-Vision-Instruct

<details>
<summary>Complete evidence: meta-llama/Llama-3.2-11B-Vision-Instruct</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 2.19s
- *Generation time:* 27.41s
- *Total time:* 29.61s
- *Input validation time:* 0.00201
- *Prompt preparation time:* 0.00511
- *First-token latency:* 1.68
- *Cleanup time:* 0.0983
- *Prompt tokens:* 15
- *Generation tokens:* 130
- *Total tokens:* 145
- *Prompt throughput (raw):* 8.92 tok/s
- *Generation throughput (raw):* 5.06 tok/s
- *Peak memory:* 25
- *Active memory:* 21
- *Cache memory:* 3.4
- *Model-load active memory:* 21.3
- *Post-cleanup active memory:* 0.00174
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 144
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
The image shows two cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a small, fluffy kitten with a long tail and a green collar, while the cat on the right is a larger, tabby cat with a short tail. Both cats are lying on their backs, with their paws stretched out to the sides. The remote controls are placed on the couch behind the cats, with the one on the left being a white remote control and the one on the right being a silver remote control. The background of the image is a pink blanket that the cats are lying on.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a small, fluffy kitten with a long tail and a green collar, while the cat on the right is a larger, tabby cat with a short tail. Both cats are lying on their backs, with their paws stretched out to the sides. The remote controls are placed on the couch behind the cats, with the one on the left being a white remote control and the one on the right being a silver remote control. The background of the image is a pink blanket that the cats are lying on.
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
- *Model load time:* 0.99s
- *Generation time:* 0.56s
- *Total time:* 1.56s
- *Input validation time:* 0.00166
- *Prompt preparation time:* 0.000886
- *First-token latency:* 0.218
- *Cleanup time:* 0.0715
- *Prompt tokens:* 768
- *Generation tokens:* 20
- *Total tokens:* 788
- *Prompt throughput (raw):* 3,516 tok/s
- *Generation throughput (raw):* 60.9 tok/s
- *Peak memory:* 9.3
- *Active memory:* 8.3
- *Cache memory:* 0.41
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 61
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
The image shows two cats lying on a pink couch with remote controls beside them.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two cats lying on a pink couch with remote controls beside them.
```

</details>

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
- *Model load time:* 2.24s
- *Generation time:* 4.11s
- *Total time:* 6.36s
- *Input validation time:* 0.00163
- *Prompt preparation time:* 0.00656
- *First-token latency:* 0.67
- *Cleanup time:* 0.0961
- *Prompt tokens:* 439
- *Generation tokens:* 107
- *Total tokens:* 546
- *Prompt throughput (raw):* 655 tok/s
- *Generation throughput (raw):* 31.3 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.14
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.00233
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 40
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
The image features two cats lying on a pink surface, likely a couch or bed. The cat on the left is positioned with its back arched and head raised, while the cat on the right is lying flat with its head turned to the side. Both cats have a tabby pattern with stripes and spots, and they appear to be relaxed and comfortable in their environment. The presence of remote controls suggests a domestic setting, possibly a living room. The cats' relaxed postures and the cozy setting indicate a peaceful and familiar atmosphere.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image features two cats lying on a pink surface, likely a couch or bed. The cat on the left is positioned with its back arched and head raised, while the cat on the right is lying flat with its head turned to the side. Both cats have a tabby pattern with stripes and spots, and they appear to be relaxed and comfortable in their environment. The presence of remote controls suggests a domestic setting, possibly a living room. The cats' relaxed postures and the cozy setting indicate a peaceful and familiar atmosphere.
```

</details>

</details>

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### mlx-community/FastVLM-0.5B-bf16

<details>
<summary>Complete evidence: mlx-community/FastVLM-0.5B-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.61s
- *Generation time:* 1.30s
- *Total time:* 1.92s
- *Input validation time:* 0.00184
- *Prompt preparation time:* 0.00161
- *First-token latency:* 0.0805
- *Cleanup time:* 0.0708
- *Prompt tokens:* 24
- *Generation tokens:* 428
- *Total tokens:* 452
- *Prompt throughput (raw):* 298 tok/s
- *Generation throughput (raw):* 360 tok/s
- *Peak memory:* 2.2
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.00287
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 135
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
The image depicts two cats lying on a pink couch. The couch is a bright pink color, and the cats are positioned side by side, facing opposite directions. The cat on the left is a tabby with a mix of black, brown, and white stripes. It is lying on its side with its head resting on the couch, appearing to be in a deep sleep. The tabby cat has a relaxed posture, with its body stretched out and its legs slightly bent.

The cat on the right is a tabby as well, but with a more distinct pattern of black and brown stripes. This cat is lying on its stomach, with its head turned towards the left cat. Its body is stretched out, and its legs are extended forward. The tabby cat's eyes are closed, indicating that it is also sleeping.

Both cats have their eyes closed, suggesting that they are either very relaxed or possibly asleep. The couch they are lying on is a bright pink color, which contrasts with the cats' fur patterns. The couch appears to be made of a soft material, likely a fabric or a velveteen, which is common for couches.

There are two remote controls placed on the couch near the cats. One remote control is white and has a standard layout with multiple buttons, while the other remote control is blue and also has a standard layout with buttons. Both remotes are placed close to the cats, suggesting that they might have been used recently and left there.

The overall scene is peaceful and cozy, with the cats appearing to be in a comfortable and safe environment. The presence of the remotes indicates that the couch is likely in a living room or a similar area where people relax and spend time. The image captures a moment of tranquility, with the cats seemingly undisturbed by their surroundings.

In summary, the image shows two tabby cats lying on a pink couch, with their eyes closed, indicating they are either sleeping or resting. The couch is a bright pink color, and there are two remotes placed on the couch near the cats. The scene conveys a sense of calm and relaxation.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image depicts two cats lying on a pink couch. The couch is a bright pink color, and the cats are positioned side by side, facing opposite directions. The cat on the left is a tabby with a mix of black, brown, and white stripes. It is lying on its side with its head resting on the couch, appearing to be in a deep sleep. The tabby cat has a relaxed posture, with its body stretched out and its legs slightly bent.

The cat on the right is a tabby as well, but with a more distinct pattern of black and brown stripes. This cat is lying on its stomach, with its head turned towards the left cat. Its body is stretched out, and its legs are extended forward. The tabby cat's eyes are closed, indicating that it is also sleeping.

Both cats have their eyes closed, suggesting that they are either very relaxed or possibly asleep. The couch they are lying on is a bright pink color, which contrasts with the cats' fur patterns. The couch appears to be made of a soft material, likely a fabric or a velveteen, which is common for couches.

There are two remote controls placed on the couch near the cats. One remote control is white and has a standard layout with multiple buttons, while the other remote control is blue and also has a standard layout with buttons. Both remotes are placed close to the cats, suggesting that they might have been used recently and left there.

The overall scene is peaceful and cozy, with the cats appearing to be in a comfortable and safe environment. The presence of the remotes indicates that the couch is likely in a living room or a similar area where people relax and spend time. The image captures a moment of tranquility, with the cats seemingly undisturbed by their surroundings.

In summary, the image shows two tabby cats lying on a pink couch, with their eyes closed, indicating they are either sleeping or resting. The couch is a bright pink color, and there are two remotes placed on the couch near the cats. The scene conveys a sense of calm and relaxation.
```

</details>

</details>

---

<a id="model-mlx-community-glm-41v-9b-thinking-8bit"></a>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

<details>
<summary>Complete evidence: mlx-community/GLM-4.1V-9B-Thinking-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 1.64s
- *Generation time:* 6.24s
- *Total time:* 7.89s
- *Input validation time:* 0.00172
- *Prompt preparation time:* 0.00162
- *First-token latency:* 0.356
- *Cleanup time:* 0.0849
- *Prompt tokens:* 402
- *Generation tokens:* 301
- *Total tokens:* 703
- *Prompt throughput (raw):* 1,130 tok/s
- *Generation throughput (raw):* 51.3 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.04
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.00318
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 97
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
&lt;think&gt;Got it, let's describe the image. First, there are two cats lying on a pink blanket, which is on a red couch. The cat on the left is a smaller one, maybe a kitten, with a mix of brown and black stripes, lying on its side with eyes closed, looking relaxed. The cat on the right is larger, also with tabby stripes, lying on its side, maybe looking down or resting. There are two remote controls: one near the left cat is a light blue remote, and the other near the right cat is a white remote with colorful buttons. The setting is cozy, with the pink blanket and red couch creating a warm atmosphere. The cats seem comfortable and relaxed, maybe napping. I need to make sure to include all elements: the cats' positions, colors, the blanket, the remotes, and the couch.&lt;/think&gt;&lt;answer&gt;The image shows two tabby cats resting on a bright pink blanket that is placed on a red couch. The cat on the left is smaller, possibly a kitten, with a mix of brown and black striped fur, lying on its side with its eyes closed, appearing relaxed. The larger cat on the right has similar tabby markings and is also lying on its side, possibly looking downward. Near the left cat is a light blue remote control, and near the right cat is a white remote with colorful buttons. The scene conveys a cozy, relaxed atmosphere, with the cats seemingly at ease in their environment.
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's describe the image. First, there are two cats lying on a pink blanket, which is on a red couch. The cat on the left is a smaller one, maybe a kitten, with a mix of brown and black stripes, lying on its side with eyes closed, looking relaxed. The cat on the right is larger, also with tabby stripes, lying on its side, maybe looking down or resting. There are two remote controls: one near the left cat is a light blue remote, and the other near the right cat is a white remote with colorful buttons. The setting is cozy, with the pink blanket and red couch creating a warm atmosphere. The cats seem comfortable and relaxed, maybe napping. I need to make sure to include all elements: the cats' positions, colors, the blanket, the remotes, and the couch.</think><answer>The image shows two tabby cats resting on a bright pink blanket that is placed on a red couch. The cat on the left is smaller, possibly a kitten, with a mix of brown and black striped fur, lying on its side with its eyes closed, appearing relaxed. The larger cat on the right has similar tabby markings and is also lying on its side, possibly looking downward. Near the left cat is a light blue remote control, and near the right cat is a white remote with colorful buttons. The scene conveys a cozy, relaxed atmosphere, with the cats seemingly at ease in their environment.
```

</details>

</details>

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### mlx-community/GLM-4.6V-Flash-6bit

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-6bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 1.40s
- *Generation time:* 3.45s
- *Total time:* 4.86s
- *Input validation time:* 0.0018
- *Prompt preparation time:* 0.00665
- *First-token latency:* 0.338
- *Cleanup time:* 0.0849
- *Prompt tokens:* 406
- *Generation tokens:* 197
- *Total tokens:* 603
- *Prompt throughput (raw):* 1,200 tok/s
- *Generation throughput (raw):* 63.6 tok/s
- *Peak memory:* 10
- *Active memory:* 9.4
- *Cache memory:* 0.05
- *Model-load active memory:* 9.43
- *Post-cleanup active memory:* 0.00351
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 121
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
The image depicts two tabby cats resting on a bright pink couch. The couch has a smooth, slightly wrinkled fabric, and the backrest is a deeper red.

- **Left Cat**: A smaller tabby with a mix of brown, black, and tan stripes. It lies on its side, eyes closed, appearing relaxed. Its tail is curled at the end, and a green collar is visible around its neck.

- **Right Cat**: A larger tabby with similar brown, black, and tan striping. It lies on its side, head lowered toward the couch, eyes partially closed, also appearing relaxed.

- **Remotes**: Two remote controls are placed on the couch between the cats. The left remote is light blue with dark buttons, while the right remote is white with colorful buttons (red, green, blue, yellow).

The overall scene is cozy and relaxed, with the cats appearing comfortable and at ease on the vibrant pink couch.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image depicts two tabby cats resting on a bright pink couch. The couch has a smooth, slightly wrinkled fabric, and the backrest is a deeper red.  

- **Left Cat**: A smaller tabby with a mix of brown, black, and tan stripes. It lies on its side, eyes closed, appearing relaxed. Its tail is curled at the end, and a green collar is visible around its neck.  

- **Right Cat**: A larger tabby with similar brown, black, and tan striping. It lies on its side, head lowered toward the couch, eyes partially closed, also appearing relaxed.  

- **Remotes**: Two remote controls are placed on the couch between the cats. The left remote is light blue with dark buttons, while the right remote is white with colorful buttons (red, green, blue, yellow).  

The overall scene is cozy and relaxed, with the cats appearing comfortable and at ease on the vibrant pink couch.
```

</details>

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
- *Model load time:* 1.28s
- *Generation time:* 2.54s
- *Total time:* 3.82s
- *Input validation time:* 0.00171
- *Prompt preparation time:* 0.000213
- *First-token latency:* 0.297
- *Cleanup time:* 0.079
- *Prompt tokens:* 406
- *Generation tokens:* 205
- *Total tokens:* 611
- *Prompt throughput (raw):* 1,367 tok/s
- *Generation throughput (raw):* 91.8 tok/s
- *Peak memory:* 7.7
- *Active memory:* 6.8
- *Cache memory:* 0.05
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00383
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 121
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
This image features two tabby cats resting on a bright pink couch.

- **Left Cat**: A slender tabby with a mix of brown, black, and white fur. It lies on its side, head resting on the couch, eyes closed in a relaxed state. Its tail is long and dark, curving slightly.

- **Right Cat**: A plump tabby with a more uniform brown and black patterned fur. It lies on its side, head lowered toward the couch, eyes partially open or closed, appearing calm. Its front paw is extended forward, while the hind leg is relaxed.

- **Remotes and Couch**: Two remote controls are visible: a light blue remote on the left side of the couch, and a white remote with colorful buttons near the right cat. The couch itself is a vibrant pink, with a smooth, slightly wrinkled fabric surface.

The overall scene conveys a sense of calm and domestic comfort, with the two cats appearing content in their shared space.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This image features two tabby cats resting on a bright pink couch.  

- **Left Cat**: A slender tabby with a mix of brown, black, and white fur. It lies on its side, head resting on the couch, eyes closed in a relaxed state. Its tail is long and dark, curving slightly.  

- **Right Cat**: A plump tabby with a more uniform brown and black patterned fur. It lies on its side, head lowered toward the couch, eyes partially open or closed, appearing calm. Its front paw is extended forward, while the hind leg is relaxed.  

- **Remotes and Couch**: Two remote controls are visible: a light blue remote on the left side of the couch, and a white remote with colorful buttons near the right cat. The couch itself is a vibrant pink, with a smooth, slightly wrinkled fabric surface.  

The overall scene conveys a sense of calm and domestic comfort, with the two cats appearing content in their shared space.
```

</details>

</details>

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### mlx-community/GLM-4.6V-nvfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-nvfp4</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 6.92s
- *Generation time:* 7.12s
- *Total time:* 14.05s
- *Input validation time:* 0.00161
- *Prompt preparation time:* 0.00685
- *First-token latency:* 3.83
- *Cleanup time:* 0.228
- *Prompt tokens:* 406
- *Generation tokens:* 172
- *Total tokens:* 578
- *Prompt throughput (raw):* 106 tok/s
- *Generation throughput (raw):* 52.5 tok/s
- *Peak memory:* 63
- *Active memory:* 62
- *Cache memory:* 0.16
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00416
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 121
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
The image shows two tabby cats resting on a bright pink blanket spread over a red couch. The cat on the is lying on its side, with its head turned slightly downward and eyes closed, appearing relaxed. Its fur displays a mix of brown, black, and tan stripes, and it has a pinkish underbelly. The cat on the left is stretched out in a more elongated position, also with eyes closed, and its body angled toward the right. This cat has a similar striped pattern but with a lighter, more muted color palette, and it wears a small green collar. Between them, two remote controls lie on the pink fabric: one is light blue with numerous buttons, and the other is white with colorful buttons. The overall scene conveys a calm, cozy atmosphere, with the cats seemingly at ease in their shared resting spot.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two tabby cats resting on a bright pink blanket spread over a red couch. The cat on the is lying on its side, with its head turned slightly downward and eyes closed, appearing relaxed. Its fur displays a mix of brown, black, and tan stripes, and it has a pinkish underbelly. The cat on the left is stretched out in a more elongated position, also with eyes closed, and its body angled toward the right. This cat has a similar striped pattern but with a lighter, more muted color palette, and it wears a small green collar. Between them, two remote controls lie on the pink fabric: one is light blue with numerous buttons, and the other is white with colorful buttons. The overall scene conveys a calm, cozy atmosphere, with the cats seemingly at ease in their shared resting spot.
```

</details>

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
- *Model load time:* 1.63s
- *Generation time:* 4.88s
- *Total time:* 6.53s
- *Input validation time:* 0.00162
- *Prompt preparation time:* 0.00177
- *First-token latency:* 2.45
- *Cleanup time:* 0.106
- *Prompt tokens:* 3,340
- *Generation tokens:* 77
- *Total tokens:* 3,417
- *Prompt throughput (raw):* 1,363 tok/s
- *Generation throughput (raw):* 32.1 tok/s
- *Peak memory:* 19
- *Active memory:* 16
- *Cache memory:* 0.78
- *Model-load active memory:* 16.4
- *Post-cleanup active memory:* 0.00474
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 45
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
The image shows two cats lying on a pink blanket on a red couch. Both cats are sleeping. The cat on the left is a tabby with a striped pattern, and the cat on the right is also a tabby with a similar pattern but with more orange tones. There are two remote controls on the couch, one near each cat. The scene appears cozy and relaxed.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two cats lying on a pink blanket on a red couch. Both cats are sleeping. The cat on the left is a tabby with a striped pattern, and the cat on the right is also a tabby with a similar pattern but with more orange tones. There are two remote controls on the couch, one near each cat. The scene appears cozy and relaxed.
```

</details>

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
- *Model load time:* 1.61s
- *Generation time:* 4.20s
- *Total time:* 5.82s
- *Input validation time:* 0.00165
- *Prompt preparation time:* 0.000181
- *First-token latency:* 1.27
- *Cleanup time:* 0.104
- *Prompt tokens:* 3,340
- *Generation tokens:* 99
- *Total tokens:* 3,439
- *Prompt throughput (raw):* 2,631 tok/s
- *Generation throughput (raw):* 34.1 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.26
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.00505
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 45
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
The image shows two cats sleeping on a pink blanket. The cat on the left is a smaller kitten, lying on its side with its head resting on the blanket. The cat on the right is a larger adult cat, also lying on its side with its head resting on the blanket. Both cats have striped fur patterns. There are two remote controls placed near the cats, one on the left side and one on the right side of the blanket. The background appears to be a red couch.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two cats sleeping on a pink blanket. The cat on the left is a smaller kitten, lying on its side with its head resting on the blanket. The cat on the right is a larger adult cat, also lying on its side with its head resting on the blanket. Both cats have striped fur patterns. There are two remote controls placed near the cats, one on the left side and one on the right side of the blanket. The background appears to be a red couch.
```

</details>

</details>

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### mlx-community/LFM2-VL-1.6B-8bit

<details>
<summary>Complete evidence: mlx-community/LFM2-VL-1.6B-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.56s
- *Generation time:* 0.40s
- *Total time:* 0.97s
- *Input validation time:* 0.00201
- *Prompt preparation time:* 0.00113
- *First-token latency:* 0.0679
- *Cleanup time:* 0.0673
- *Prompt tokens:* 267
- *Generation tokens:* 103
- *Total tokens:* 370
- *Prompt throughput (raw):* 3,931 tok/s
- *Generation throughput (raw):* 330 tok/s
- *Peak memory:* 3.0
- *Active memory:* 2.1
- *Cache memory:* 0.02
- *Model-load active memory:* 2.06
- *Post-cleanup active memory:* 0.00649
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 91
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
The image depicts two cats lying on a bright pink blanket, which is spread over a red couch. The cat on the left is a gray tabby with black stripes, while the cat on the right is a brown tabby with black stripes. Both cats are stretched out and appear to be sleeping. The pink blanket is slightly wrinkled, and there are two white remote controls placed on it. The scene is set indoors, and the vibrant colors of the blanket and the cats' fur create a striking contrast.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image depicts two cats lying on a bright pink blanket, which is spread over a red couch. The cat on the left is a gray tabby with black stripes, while the cat on the right is a brown tabby with black stripes. Both cats are stretched out and appear to be sleeping. The pink blanket is slightly wrinkled, and there are two white remote controls placed on it. The scene is set indoors, and the vibrant colors of the blanket and the cats' fur create a striking contrast.
```

</details>

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
- *Model load time:* 0.58s
- *Generation time:* 0.68s
- *Total time:* 1.26s
- *Input validation time:* 0.00179
- *Prompt preparation time:* 0.00333
- *First-token latency:* 0.0683
- *Cleanup time:* 0.0673
- *Prompt tokens:* 267
- *Generation tokens:* 108
- *Total tokens:* 375
- *Prompt throughput (raw):* 3,907 tok/s
- *Generation throughput (raw):* 186 tok/s
- *Peak memory:* 4.1
- *Active memory:* 3.2
- *Cache memory:* 0.01
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.00662
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 91
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
The image depicts two cats lying on a pink blanket spread over a red couch. The cat on the left is a tabby with black stripes and a green collar, while the cat on the right is a calico with a mix of brown, black, and white fur. Both cats are sleeping peacefully, with their eyes closed and their bodies stretched out. In the background, there are two remote controls placed on the couch. The scene is cozy and serene, capturing a moment of relaxation for the two feline companions.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image depicts two cats lying on a pink blanket spread over a red couch. The cat on the left is a tabby with black stripes and a green collar, while the cat on the right is a calico with a mix of brown, black, and white fur. Both cats are sleeping peacefully, with their eyes closed and their bodies stretched out. In the background, there are two remote controls placed on the couch. The scene is cozy and serene, capturing a moment of relaxation for the two feline companions.
```

</details>

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
- *Model load time:* 1.46s
- *Generation time:* 10.75s
- *Total time:* 12.22s
- *Input validation time:* 0.00156
- *Prompt preparation time:* 0.00131
- *First-token latency:* 1.4
- *Cleanup time:* 0.0895
- *Prompt tokens:* 14
- *Generation tokens:* 155
- *Total tokens:* 169
- *Prompt throughput (raw):* 10.0 tok/s
- *Generation throughput (raw):* 16.6 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.52
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.00714
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 144
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
The image shows two cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a small, fluffy kitten with a long tail and a green collar. The cat on the right is a larger, tabby cat with a striped coat and a short tail. Both cats are lying on their backs, with their paws stretched out to the sides. The remote controls are placed on the couch behind the cats, with the one on the left being a white remote control and the one on the right being a gray remote control. The background of the image is a pink blanket that the cats are lying on. Overall, the image appears to be a casual and relaxed scene, with the cats enjoying a peaceful moment together.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a small, fluffy kitten with a long tail and a green collar. The cat on the right is a larger, tabby cat with a striped coat and a short tail. Both cats are lying on their backs, with their paws stretched out to the sides. The remote controls are placed on the couch behind the cats, with the one on the left being a white remote control and the one on the right being a gray remote control. The background of the image is a pink blanket that the cats are lying on. Overall, the image appears to be a casual and relaxed scene, with the cats enjoying a peaceful moment together.
```

</details>

</details>

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### mlx-community/MiniCPM-V-4.6-8bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.6-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.89s
- *Generation time:* 0.53s
- *Total time:* 1.44s
- *Input validation time:* 0.00178
- *Prompt preparation time:* 0.00795
- *First-token latency:* 0.105
- *Cleanup time:* 0.0795
- *Prompt tokens:* 226
- *Generation tokens:* 112
- *Total tokens:* 338
- *Prompt throughput (raw):* 2,145 tok/s
- *Generation throughput (raw):* 285 tok/s
- *Peak memory:* 3.0
- *Active memory:* 2.3
- *Cache memory:* 0.05
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00767
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 108
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
The image shows two tabby cats resting comfortably on a bright pink blanket or couch. Both cats appear to be asleep or deeply relaxed, with their bodies stretched out. One cat is on the left side, lying with its legs extended and a remote control placed nearby. The other cat is on the right, also in a relaxed position, with its paws slightly spread. The background includes a red or dark-colored couch, adding a cozy and warm atmosphere to the scene. The overall mood is peaceful and tranquil, capturing a moment of rest for the cats.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two tabby cats resting comfortably on a bright pink blanket or couch. Both cats appear to be asleep or deeply relaxed, with their bodies stretched out. One cat is on the left side, lying with its legs extended and a remote control placed nearby. The other cat is on the right, also in a relaxed position, with its paws slightly spread. The background includes a red or dark-colored couch, adding a cozy and warm atmosphere to the scene. The overall mood is peaceful and tranquil, capturing a moment of rest for the cats.
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
- *Model load time:* 1.32s
- *Generation time:* 2.92s
- *Total time:* 4.26s
- *Input validation time:* 0.00175
- *Prompt preparation time:* 0.00648
- *First-token latency:* 0.665
- *Cleanup time:* 0.0831
- *Prompt tokens:* 972
- *Generation tokens:* 155
- *Total tokens:* 1,127
- *Prompt throughput (raw):* 1,461 tok/s
- *Generation throughput (raw):* 69.1 tok/s
- *Peak memory:* 9.8
- *Active memory:* 8.0
- *Cache memory:* 0.23
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.00793
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 2478
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
This image shows two cats lounging on a pink blanket or cushion. Both cats are lying on their backs, with their legs stretched out in a relaxed pose.

- The cat on the left appears to be a smaller kitten with a tabby pattern (striped fur with swirling patches). It has its head tilted slightly to the side and is looking upward.
- The cat on the right is larger and also has a tabby pattern. It is lying with its head resting on the blanket and appears to be looking off to the side.

In the background, there are two remote controls placed on either side of the cats, suggesting a cozy indoor setting, possibly on a couch or bed. The overall mood of the image is one of comfort and relaxation.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This image shows two cats lounging on a pink blanket or cushion. Both cats are lying on their backs, with their legs stretched out in a relaxed pose.

- The cat on the left appears to be a smaller kitten with a tabby pattern (striped fur with swirling patches). It has its head tilted slightly to the side and is looking upward.
- The cat on the right is larger and also has a tabby pattern. It is lying with its head resting on the blanket and appears to be looking off to the side.

In the background, there are two remote controls placed on either side of the cats, suggesting a cozy indoor setting, possibly on a couch or bed. The overall mood of the image is one of comfort and relaxation.
```

</details>

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
- *Model load time:* 1.43s
- *Generation time:* 3.25s
- *Total time:* 4.68s
- *Input validation time:* 0.00161
- *Prompt preparation time:* 0.00025
- *First-token latency:* 0.76
- *Cleanup time:* 0.0824
- *Prompt tokens:* 972
- *Generation tokens:* 156
- *Total tokens:* 1,128
- *Prompt throughput (raw):* 1,280 tok/s
- *Generation throughput (raw):* 63.0 tok/s
- *Peak memory:* 10
- *Active memory:* 8.4
- *Cache memory:* 0.23
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00819
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 2478
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
This image features two tabby cats lounging on a pink cushion, likely part of a couch. Both cats appear relaxed and comfortable:

- **Left Cat (smaller one):** Stretched out on its side, with its front legs extended forward and its head resting on its paws. Its fur pattern includes dark stripes and lighter patches, typical of a tabby.

- **Right Cat (larger one):** Also lying on its side, facing slightly towards its companion. Its body is elongated, and its head rests on its front paws. Its fur pattern mirrors that of the smaller cat, with similar tabby markings.

Both cats seem to be enjoying their time, perhaps watching something on a television, given their proximity to remote controls placed on either side of them.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This image features two tabby cats lounging on a pink cushion, likely part of a couch. Both cats appear relaxed and comfortable:

- **Left Cat (smaller one):** Stretched out on its side, with its front legs extended forward and its head resting on its paws. Its fur pattern includes dark stripes and lighter patches, typical of a tabby.

- **Right Cat (larger one):** Also lying on its side, facing slightly towards its companion. Its body is elongated, and its head rests on its front paws. Its fur pattern mirrors that of the smaller cat, with similar tabby markings.

Both cats seem to be enjoying their time, perhaps watching something on a television, given their proximity to remote controls placed on either side of them.
```

</details>

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
- *Model load time:* 0.93s
- *Generation time:* 1.27s
- *Total time:* 2.21s
- *Input validation time:* 0.00163
- *Prompt preparation time:* 0.00635
- *First-token latency:* 0.281
- *Cleanup time:* 0.0717
- *Prompt tokens:* 971
- *Generation tokens:* 178
- *Total tokens:* 1,149
- *Prompt throughput (raw):* 3,456 tok/s
- *Generation throughput (raw):* 187 tok/s
- *Peak memory:* 4.5
- *Active memory:* 2.8
- *Cache memory:* 0.16
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00846
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 2477
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
This image depicts two cats lying comfortably on a soft, pink surface, likely a couch or bed.

The cat on the left appears to be a kitten, judging by its smaller size, fluffier fur, and the tag around its neck, which is common for pet identification. Its fur has a mix of light and dark stripes, typical of many domestic shorthair cats.

The cat on the right is larger and seems to be an adult, given its more developed size and mature appearance. Its fur is also striped but appears denser and more textured, with a mix of light and dark bands. The adult cat is lying on its side, with its head slightly raised and looking relaxed.

Both cats are positioned near what look like remote controls, suggesting a cozy, indoor setting where they are resting. The scene exudes a sense of calm and companionship.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This image depicts two cats lying comfortably on a soft, pink surface, likely a couch or bed.

The cat on the left appears to be a kitten, judging by its smaller size, fluffier fur, and the tag around its neck, which is common for pet identification. Its fur has a mix of light and dark stripes, typical of many domestic shorthair cats.

The cat on the right is larger and seems to be an adult, given its more developed size and mature appearance. Its fur is also striped but appears denser and more textured, with a mix of light and dark bands. The adult cat is lying on its side, with its head slightly raised and looking relaxed.

Both cats are positioned near what look like remote controls, suggesting a cozy, indoor setting where they are resting. The scene exudes a sense of calm and companionship.
```

</details>

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
- *Model load time:* 1.29s
- *Generation time:* 3.68s
- *Total time:* 4.98s
- *Input validation time:* 0.00156
- *Prompt preparation time:* 0.00132
- *First-token latency:* 0.399
- *Cleanup time:* 0.0836
- *Prompt tokens:* 747
- *Generation tokens:* 174
- *Total tokens:* 921
- *Prompt throughput (raw):* 1,872 tok/s
- *Generation throughput (raw):* 53.5 tok/s
- *Peak memory:* 11
- *Active memory:* 9.0
- *Cache memory:* 0.08
- *Model-load active memory:* 9.05
- *Post-cleanup active memory:* 0.00877
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 36
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
 In this detailed image, two cats are peacefully sleeping on a red couch, which is covered with a pink blanket. The cat on the left is a slender, gray and black striped feline with a white belly and a black tail. It is lying on its side with its front paws stretched out and its head resting on its front paws. A green collar adorns its neck, and a white remote control is positioned behind its head. The cat on the right is a larger, chunkier cat with a mix of brown, black, and white fur, featuring a distinctive white spot on its belly. This cat is lying on its side with its head down and its back legs stretched out. Between the two cats, another white remote control is visible. The scene is cozy and serene, with the cats comfortably nestled on the pink blanket against the red couch.
</pre>

<details>
<summary>Exact raw output</summary>

```text
 In this detailed image, two cats are peacefully sleeping on a red couch, which is covered with a pink blanket. The cat on the left is a slender, gray and black striped feline with a white belly and a black tail. It is lying on its side with its front paws stretched out and its head resting on its front paws. A green collar adorns its neck, and a white remote control is positioned behind its head. The cat on the right is a larger, chunkier cat with a mix of brown, black, and white fur, featuring a distinctive white spot on its belly. This cat is lying on its side with its head down and its back legs stretched out. Between the two cats, another white remote control is visible. The scene is cozy and serene, with the cats comfortably nestled on the pink blanket against the red couch.
```

</details>

</details>

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### mlx-community/Molmo-7B-D-0924-bf16

<details>
<summary>Complete evidence: mlx-community/Molmo-7B-D-0924-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 1.68s
- *Generation time:* 6.89s
- *Total time:* 8.58s
- *Input validation time:* 0.00158
- *Prompt preparation time:* 0.00123
- *First-token latency:* 0.424
- *Cleanup time:* 0.105
- *Prompt tokens:* 747
- *Generation tokens:* 199
- *Total tokens:* 946
- *Prompt throughput (raw):* 1,760 tok/s
- *Generation throughput (raw):* 30.9 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.08
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.00908
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 36
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
 In this detailed image, two cats are peacefully sleeping on a red couch, which is covered with a pink blanket. The cat on the left is a slender, gray and black striped feline with a white belly and a black tail. It is lying on its side with its front paws stretched out and its back legs tucked in. This cat is wearing a green collar and has its eyes closed, with its head slightly turned to the right. Behind this cat, there is a white remote control. The cat on the right is a larger, chunkier cat with a mix of brown, black, and white fur, and a white belly. It is also lying on its side, with its head facing the camera and its back legs stretched out. This cat has its eyes closed and its front paws tucked in. Between the two cats, there is another white remote control. The couch itself is a deep red color, and the overall scene exudes a sense of calm and comfort.
</pre>

<details>
<summary>Exact raw output</summary>

```text
 In this detailed image, two cats are peacefully sleeping on a red couch, which is covered with a pink blanket. The cat on the left is a slender, gray and black striped feline with a white belly and a black tail. It is lying on its side with its front paws stretched out and its back legs tucked in. This cat is wearing a green collar and has its eyes closed, with its head slightly turned to the right. Behind this cat, there is a white remote control. The cat on the right is a larger, chunkier cat with a mix of brown, black, and white fur, and a white belly. It is also lying on its side, with its head facing the camera and its back legs stretched out. This cat has its eyes closed and its front paws tucked in. Between the two cats, there is another white remote control. The couch itself is a deep red color, and the overall scene exudes a sense of calm and comfort.
```

</details>

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
- *Model load time:* 2.22s
- *Generation time:* 27.46s
- *Total time:* 29.69s
- *Input validation time:* 0.00164
- *Prompt preparation time:* 0.00487
- *First-token latency:* 0.765
- *Cleanup time:* 0.0965
- *Prompt tokens:* 764
- *Generation tokens:* 158
- *Total tokens:* 922
- *Prompt throughput (raw):* 999 tok/s
- *Generation throughput (raw):* 5.95 tok/s
- *Peak memory:* 23
- *Active memory:* 19
- *Cache memory:* 4.3
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 0.0097
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 78
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
In this detailed photograph, two cats are peacefully sleeping on a pink blanket that covers a red couch. The cats, both with a mix of brown, black, and white fur, are positioned side by side, facing each other. The cat on the left is lying on its side with its head resting on the pink blanket, while the cat on the right is also on its side but with its head slightly turned towards the camera. Both cats have their eyes closed, indicating they are deeply asleep. The cat on the left has a green collar around its neck. Behind the cats, there are two remote controls: one is white with blue and red buttons, and the other is gray with blue and red buttons. The overall scene is cozy and serene, capturing a moment of tranquility.
</pre>

<details>
<summary>Exact raw output</summary>

```text
In this detailed photograph, two cats are peacefully sleeping on a pink blanket that covers a red couch. The cats, both with a mix of brown, black, and white fur, are positioned side by side, facing each other. The cat on the left is lying on its side with its head resting on the pink blanket, while the cat on the right is also on its side but with its head slightly turned towards the camera. Both cats have their eyes closed, indicating they are deeply asleep. The cat on the left has a green collar around its neck. Behind the cats, there are two remote controls: one is white with blue and red buttons, and the other is gray with blue and red buttons. The overall scene is cozy and serene, capturing a moment of tranquility.
```

</details>

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
- *Model load time:* 10.05s
- *Generation time:* 10.21s
- *Total time:* 20.28s
- *Input validation time:* 0.00164
- *Prompt preparation time:* 0.0133
- *First-token latency:* 5.67
- *Cleanup time:* 0.279
- *Prompt tokens:* 317
- *Generation tokens:* 305
- *Total tokens:* 622
- *Prompt throughput (raw):* 55.9 tok/s
- *Generation throughput (raw):* 67.4 tok/s
- *Peak memory:* 71
- *Active memory:* 70
- *Cache memory:* 0.10
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 131
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
This is a cozy, overhead photograph of two tabby cats sleeping peacefully on a vibrant pink couch.

**The Cats:**
- **Left Cat:** A smaller tabby cat with classic brown and black stripes lies stretched out on its side, facing right. Its tail curls near its hind legs, and one front paw is extended forward. It appears to be in a deep sleep.
- **Right Cat:** A larger tabby cat, possibly the same breed or mix, lies curled up more compactly on its side, facing left. Its head is tucked down, and its body forms a gentle arc. The fur shows rich brown, black, and tan markings.

**The Setting:**
- They are resting on a bright, fuchsia-pink fabric — likely a blanket or the couch upholstery itself — which creates a striking contrast with their earth-toned fur.
- Two remote controls are visible:
  - One white remote with blue buttons lies near the head of the smaller cat.
  - Another white remote, slightly longer and with green and blue buttons, is positioned between the two cats.

**Overall Mood:**
The image exudes warmth, comfort, and tranquility. The cats look completely relaxed and content in their shared space, suggesting a close bond between them. The bold pink background adds a playful, cheerful touch to the serene scene.

It’s a charming snapshot of domestic life — cats napping together on a colorful sofa, surrounded by the tools of human entertainment.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This is a cozy, overhead photograph of two tabby cats sleeping peacefully on a vibrant pink couch.

**The Cats:**
- **Left Cat:** A smaller tabby cat with classic brown and black stripes lies stretched out on its side, facing right. Its tail curls near its hind legs, and one front paw is extended forward. It appears to be in a deep sleep.
- **Right Cat:** A larger tabby cat, possibly the same breed or mix, lies curled up more compactly on its side, facing left. Its head is tucked down, and its body forms a gentle arc. The fur shows rich brown, black, and tan markings.

**The Setting:**
- They are resting on a bright, fuchsia-pink fabric — likely a blanket or the couch upholstery itself — which creates a striking contrast with their earth-toned fur.
- Two remote controls are visible:
  - One white remote with blue buttons lies near the head of the smaller cat.
  - Another white remote, slightly longer and with green and blue buttons, is positioned between the two cats.

**Overall Mood:**
The image exudes warmth, comfort, and tranquility. The cats look completely relaxed and content in their shared space, suggesting a close bond between them. The bold pink background adds a playful, cheerful touch to the serene scene.

It’s a charming snapshot of domestic life — cats napping together on a colorful sofa, surrounded by the tools of human entertainment.
```

</details>

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
- *Model load time:* 1.00s
- *Generation time:* 0.58s
- *Total time:* 1.58s
- *Input validation time:* 0.00474
- *Prompt preparation time:* 0.000826
- *First-token latency:* 0.238
- *Cleanup time:* 0.0806
- *Prompt tokens:* 768
- *Generation tokens:* 20
- *Total tokens:* 788
- *Prompt throughput (raw):* 3,228 tok/s
- *Generation throughput (raw):* 61.9 tok/s
- *Peak memory:* 9.3
- *Active memory:* 8.3
- *Cache memory:* 0.41
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 61
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
The image shows two cats lying on a pink couch with remote controls beside them.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two cats lying on a pink couch with remote controls beside them.
```

</details>

</details>

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### mlx-community/Qwen2-VL-2B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen2-VL-2B-Instruct-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.52s
- *Generation time:* 0.37s
- *Total time:* 0.90s
- *Input validation time:* 0.00152
- *Prompt preparation time:* 0.0033
- *First-token latency:* 0.14
- *Cleanup time:* 0.0733
- *Prompt tokens:* 415
- *Generation tokens:* 69
- *Total tokens:* 484
- *Prompt throughput (raw):* 2,960 tok/s
- *Generation throughput (raw):* 331 tok/s
- *Peak memory:* 2.5
- *Active memory:* 1.3
- *Cache memory:* 0.03
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.0106
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 170
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
In the image, there are two cats lying on a pink blanket. One cat is on the left side, and the other cat is on the right side. Both cats appear to be sleeping or resting. There are two remote controls placed on the blanket, one on the left side and the other on the right side, near the cats.
</pre>

<details>
<summary>Exact raw output</summary>

```text
In the image, there are two cats lying on a pink blanket. One cat is on the left side, and the other cat is on the right side. Both cats appear to be sleeping or resting. There are two remote controls placed on the blanket, one on the left side and the other on the right side, near the cats.
```

</details>

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
- *Model load time:* 0.86s
- *Generation time:* 2.25s
- *Total time:* 3.12s
- *Input validation time:* 0.00155
- *Prompt preparation time:* 0.000204
- *First-token latency:* 0.0824
- *Cleanup time:* 0.078
- *Prompt tokens:* 313
- *Generation tokens:* 281
- *Total tokens:* 594
- *Prompt throughput (raw):* 3,797 tok/s
- *Generation throughput (raw):* 132 tok/s
- *Peak memory:* 5.3
- *Active memory:* 4.3
- *Cache memory:* 0.10
- *Model-load active memory:* 4.27
- *Post-cleanup active memory:* 0.0109
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 112
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
This is a photograph of two cats resting on a bright pink couch. The scene is calm and cozy, with the cats appearing to be asleep or very relaxed.

- **The Cats**: There are two cats, both with tabby patterns.
    - The cat on the left is a smaller, younger-looking tabby with a mix of black, brown, and grey stripes. It is lying on its back with its paws stretched out, and its eyes are closed.
    - The cat on the right is larger and has a more prominent brown and black striped pattern. It is lying on its side, also with its eyes closed, and appears to be sleeping peacefully.

- **The Setting**: The cats are on a vibrant pink couch, which is the main background of the image. The couch is covered with a soft-looking fabric.

- **The Remotes**: Two remote controls are visible on the couch, placed near the cats.
    - One is a light blue or grey remote control, and the other is a white remote control with a blue button.
    - The remotes are likely for a television or a similar electronic device.

- **Overall Impression**: The image captures a peaceful moment of two cats resting together on a colorful couch. The bright pink color of the couch contrasts with the cats' fur, making them stand out. The overall mood is one of tranquility and comfort.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This is a photograph of two cats resting on a bright pink couch. The scene is calm and cozy, with the cats appearing to be asleep or very relaxed.

- **The Cats**: There are two cats, both with tabby patterns.
    - The cat on the left is a smaller, younger-looking tabby with a mix of black, brown, and grey stripes. It is lying on its back with its paws stretched out, and its eyes are closed.
    - The cat on the right is larger and has a more prominent brown and black striped pattern. It is lying on its side, also with its eyes closed, and appears to be sleeping peacefully.

- **The Setting**: The cats are on a vibrant pink couch, which is the main background of the image. The couch is covered with a soft-looking fabric.

- **The Remotes**: Two remote controls are visible on the couch, placed near the cats.
    - One is a light blue or grey remote control, and the other is a white remote control with a blue button.
    - The remotes are likely for a television or a similar electronic device.

- **Overall Impression**: The image captures a peaceful moment of two cats resting together on a colorful couch. The bright pink color of the couch contrasts with the cats' fur, making them stand out. The overall mood is one of tranquility and comfort.
```

</details>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.83s
- *Generation time:* 3.19s
- *Total time:* 4.03s
- *Input validation time:* 0.00173
- *Prompt preparation time:* 0.0109
- *First-token latency:* 0.0825
- *Cleanup time:* 0.0767
- *Prompt tokens:* 315
- *Generation tokens:* 401
- *Total tokens:* 716
- *Prompt throughput (raw):* 3,817 tok/s
- *Generation throughput (raw):* 131 tok/s
- *Peak memory:* 5.3
- *Active memory:* 4.3
- *Cache memory:* 0.10
- *Model-load active memory:* 4.27
- *Post-cleanup active memory:* 0.0112
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 120
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
So, let's see. The image shows two cats lying on a bright pink couch. First, I need to describe the main elements. There are two cats, both with tabby patterns—striped fur. The cat on the left has a mix of brown, black, and maybe some gray, with a green tag on its ear. The cat on the right is also tabby, with more orange and brown tones. Both are lying down, relaxed, probably sleeping or resting. Then there are two remote controls: one is a gray remote with lots of buttons, and the other is a white remote with some blue and green buttons. The couch is a vivid pink, so that's a key color. The background is the couch, which is the main surface they're lying on. Let me check the details: the cats' positions, the remotes, the color of the couch. Also, the lighting—since it's a photo, it's probably taken in a well-lit room. Need to make sure to mention the cats' posture, the remotes, the couch color, and the overall scene. Let's structure it: start with the setting (pink couch), then the two cats, their features, then the remotes. Make sure to be descriptive but concise.
&lt;/think&gt;

The image depicts two tabby cats resting on a vibrant pink couch. Both cats are lying on their sides, appearing relaxed and asleep. The cat on the left has a mix of brown, black, and gray fur with a green ear tag, while the cat on the right features a more orange-brown tabby pattern. Between them, two remote controls are placed: a gray remote with numerous buttons and a white remote with blue and green buttons. The couch’s bright pink fabric serves as a vivid backdrop, contrasting with the cats’ striped fur and the neutral tones of the remotes. The scene is calm and cozy, capturing a peaceful moment of feline rest in a home setting.
</pre>

<details>
<summary>Exact raw output</summary>

```text
So, let's see. The image shows two cats lying on a bright pink couch. First, I need to describe the main elements. There are two cats, both with tabby patterns—striped fur. The cat on the left has a mix of brown, black, and maybe some gray, with a green tag on its ear. The cat on the right is also tabby, with more orange and brown tones. Both are lying down, relaxed, probably sleeping or resting. Then there are two remote controls: one is a gray remote with lots of buttons, and the other is a white remote with some blue and green buttons. The couch is a vivid pink, so that's a key color. The background is the couch, which is the main surface they're lying on. Let me check the details: the cats' positions, the remotes, the color of the couch. Also, the lighting—since it's a photo, it's probably taken in a well-lit room. Need to make sure to mention the cats' posture, the remotes, the couch color, and the overall scene. Let's structure it: start with the setting (pink couch), then the two cats, their features, then the remotes. Make sure to be descriptive but concise.
</think>

The image depicts two tabby cats resting on a vibrant pink couch. Both cats are lying on their sides, appearing relaxed and asleep. The cat on the left has a mix of brown, black, and gray fur with a green ear tag, while the cat on the right features a more orange-brown tabby pattern. Between them, two remote controls are placed: a gray remote with numerous buttons and a white remote with blue and green buttons. The couch’s bright pink fabric serves as a vivid backdrop, contrasting with the cats’ striped fur and the neutral tones of the remotes. The scene is calm and cozy, capturing a peaceful moment of feline rest in a home setting.
```

</details>

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
- *Model load time:* 2.21s
- *Generation time:* 10.08s
- *Total time:* 12.32s
- *Input validation time:* 0.00169
- *Prompt preparation time:* 0.0139
- *First-token latency:* 0.604
- *Cleanup time:* 0.11
- *Prompt tokens:* 317
- *Generation tokens:* 315
- *Total tokens:* 632
- *Prompt throughput (raw):* 525 tok/s
- *Generation throughput (raw):* 33.3 tok/s
- *Peak memory:* 19
- *Active memory:* 18
- *Cache memory:* 0.24
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0117
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 131
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
This is a cozy, overhead photograph of two tabby cats lounging on a vibrant pink blanket that covers a red couch or sofa.

- **The Cat on the Left**: This is a smaller, younger-looking tabby cat with classic black and brown stripes over a lighter tan coat. It’s lying on its back in a relaxed, playful pose, with its paws curled up and its head tilted to the side. A small green collar is visible around its neck. Its eyes are closed, suggesting it’s asleep or very content.

- **The Cat on the Right**: This is a larger, adult tabby cat with similar striping but more prominent orange-brown tones mixed with black. It’s curled up on its side, facing away from the camera, with its head resting near its front paws. It also appears to be sleeping peacefully.

- **The Setting**: The cats are nestled on a bright, saturated pink fleece or microfiber blanket, which contrasts vividly with the deep red fabric of the couch beneath it. Two white remote controls lie between the cats — one near each cat — adding a touch of everyday domestic life to the scene.

- **Overall Mood**: The image exudes warmth, comfort, and tranquility. The soft textures, the relaxed postures of the cats, and the bold color palette create a charming and heartwarming snapshot of feline companionship and rest.

It’s a delightful moment captured — two cats, two remotes, and a splash of pink, all sharing a quiet afternoon nap.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This is a cozy, overhead photograph of two tabby cats lounging on a vibrant pink blanket that covers a red couch or sofa.

- **The Cat on the Left**: This is a smaller, younger-looking tabby cat with classic black and brown stripes over a lighter tan coat. It’s lying on its back in a relaxed, playful pose, with its paws curled up and its head tilted to the side. A small green collar is visible around its neck. Its eyes are closed, suggesting it’s asleep or very content.

- **The Cat on the Right**: This is a larger, adult tabby cat with similar striping but more prominent orange-brown tones mixed with black. It’s curled up on its side, facing away from the camera, with its head resting near its front paws. It also appears to be sleeping peacefully.

- **The Setting**: The cats are nestled on a bright, saturated pink fleece or microfiber blanket, which contrasts vividly with the deep red fabric of the couch beneath it. Two white remote controls lie between the cats — one near each cat — adding a touch of everyday domestic life to the scene.

- **Overall Mood**: The image exudes warmth, comfort, and tranquility. The soft textures, the relaxed postures of the cats, and the bold color palette create a charming and heartwarming snapshot of feline companionship and rest.

It’s a delightful moment captured — two cats, two remotes, and a splash of pink, all sharing a quiet afternoon nap.
```

</details>

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
- *Model load time:* 3.21s
- *Generation time:* 17.41s
- *Total time:* 20.65s
- *Input validation time:* 0.00174
- *Prompt preparation time:* 0.0148
- *First-token latency:* 0.705
- *Cleanup time:* 0.129
- *Prompt tokens:* 317
- *Generation tokens:* 305
- *Total tokens:* 622
- *Prompt throughput (raw):* 450 tok/s
- *Generation throughput (raw):* 18.3 tok/s
- *Peak memory:* 30
- *Active memory:* 29
- *Cache memory:* 0.23
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.0122
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 131
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
This is a cozy, heartwarming image of two tabby cats lounging on a vibrant pink couch or blanket.

- **The Cats**:
  - On the left is a smaller, younger-looking tabby cat (possibly a kitten) with classic black and brown stripes on a lighter background. It’s stretched out on its side, appearing to be asleep or very relaxed, with one paw slightly raised.
  - On the right is a larger, adult tabby cat, also striped but with more orange/brown tones mixed in. It’s curled up slightly, facing downward, seemingly dozing off as well.
  - Both cats have similar coloring patterns, suggesting they might be related — perhaps mother and kitten.

- **The Setting**:
  - They’re lying on a bright, saturated pink fabric — likely a couch cover or throw blanket — which contrasts beautifully with their earthy fur tones.
  - Behind them is a deep red or maroon cushion or backrest of the sofa.

- **Additional Details**:
  - Two white remote controls are visible: one near the kitten’s head and another between the two cats. Their presence adds a touch of everyday domestic life to the scene.
  - The lighting is soft and even, highlighting the texture of the cats’ fur and the plushness of the fabric.

Overall, the image conveys a sense of comfort, tranquility, and feline companionship — a perfect snapshot of lazy afternoon relaxation at home.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This is a cozy, heartwarming image of two tabby cats lounging on a vibrant pink couch or blanket.

- **The Cats**:
  - On the left is a smaller, younger-looking tabby cat (possibly a kitten) with classic black and brown stripes on a lighter background. It’s stretched out on its side, appearing to be asleep or very relaxed, with one paw slightly raised.
  - On the right is a larger, adult tabby cat, also striped but with more orange/brown tones mixed in. It’s curled up slightly, facing downward, seemingly dozing off as well.
  - Both cats have similar coloring patterns, suggesting they might be related — perhaps mother and kitten.

- **The Setting**:
  - They’re lying on a bright, saturated pink fabric — likely a couch cover or throw blanket — which contrasts beautifully with their earthy fur tones.
  - Behind them is a deep red or maroon cushion or backrest of the sofa.

- **Additional Details**:
  - Two white remote controls are visible: one near the kitten’s head and another between the two cats. Their presence adds a touch of everyday domestic life to the scene.
  - The lighting is soft and even, highlighting the texture of the cats’ fur and the plushness of the fabric.

Overall, the image conveys a sense of comfort, tranquility, and feline companionship — a perfect snapshot of lazy afternoon relaxation at home.
```

</details>

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
- *Model load time:* 2.51s
- *Generation time:* 3.06s
- *Total time:* 5.59s
- *Input validation time:* 0.00179
- *Prompt preparation time:* 0.00025
- *First-token latency:* 0.378
- *Cleanup time:* 0.12
- *Prompt tokens:* 317
- *Generation tokens:* 326
- *Total tokens:* 643
- *Prompt throughput (raw):* 839 tok/s
- *Generation throughput (raw):* 122 tok/s
- *Peak memory:* 21
- *Active memory:* 21
- *Cache memory:* 0.10
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.0127
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 131
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
This is a cozy, endearing photograph of two tabby cats sleeping peacefully on a bright pink couch.

- **The Cats**: Both cats are tabbies with similar brown and black striped patterns, though the one on the right appears slightly larger and more mature. They are lying close together, almost mirror images in their relaxed poses.
    - The cat on the left is stretched out on its side, head resting near a white remote control, with its tail curled slightly. It has a small green tag on its collar.
    - The cat on the right is curled up more compactly, facing downward, with its paws tucked in. Its eyes are closed, and it looks deeply asleep.

- **The Setting**: The cats are lounging on a vibrant, fuchsia-pink fabric surface — likely a couch or blanket. The soft texture of the fabric contrasts nicely with the cats’ fur.

- **Objects**: Two remote controls are visible:
    - A larger, white remote with many buttons lies next to the left cat.
    - A smaller, slimmer white remote (possibly for a TV or cable box) is positioned between the two cats.

- **Mood &amp; Composition**: The image conveys warmth, comfort, and tranquility. The bright pink background makes the scene pop, while the cats’ peaceful slumber adds a sense of calm and domestic bliss. The composition is balanced, with the cats and remotes arranged naturally across the frame.

Overall, it’s a charming snapshot of feline relaxation — perfect for anyone who loves cats and cozy home scenes.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This is a cozy, endearing photograph of two tabby cats sleeping peacefully on a bright pink couch.

- **The Cats**: Both cats are tabbies with similar brown and black striped patterns, though the one on the right appears slightly larger and more mature. They are lying close together, almost mirror images in their relaxed poses.
    - The cat on the left is stretched out on its side, head resting near a white remote control, with its tail curled slightly. It has a small green tag on its collar.
    - The cat on the right is curled up more compactly, facing downward, with its paws tucked in. Its eyes are closed, and it looks deeply asleep.

- **The Setting**: The cats are lounging on a vibrant, fuchsia-pink fabric surface — likely a couch or blanket. The soft texture of the fabric contrasts nicely with the cats’ fur.

- **Objects**: Two remote controls are visible:
    - A larger, white remote with many buttons lies next to the left cat.
    - A smaller, slimmer white remote (possibly for a TV or cable box) is positioned between the two cats.

- **Mood & Composition**: The image conveys warmth, comfort, and tranquility. The bright pink background makes the scene pop, while the cats’ peaceful slumber adds a sense of calm and domestic bliss. The composition is balanced, with the cats and remotes arranged naturally across the frame.

Overall, it’s a charming snapshot of feline relaxation — perfect for anyone who loves cats and cozy home scenes.
```

</details>

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
- *Model load time:* 3.28s
- *Generation time:* 3.39s
- *Total time:* 6.68s
- *Input validation time:* 0.00165
- *Prompt preparation time:* 0.000244
- *First-token latency:* 0.445
- *Cleanup time:* 0.134
- *Prompt tokens:* 317
- *Generation tokens:* 299
- *Total tokens:* 616
- *Prompt throughput (raw):* 712 tok/s
- *Generation throughput (raw):* 102 tok/s
- *Peak memory:* 30
- *Active memory:* 30
- *Cache memory:* 0.10
- *Model-load active memory:* 29.1
- *Post-cleanup active memory:* 0.0132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 131
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
This is a cozy, overhead photograph of two cats sleeping on a vibrant pink blanket draped over a red couch.

- **The Cats**:
  - On the left is a smaller cat, likely a kitten or young tabby, with classic brown and black stripes. It’s stretched out on its side, paws extended, with a fluffy tail curled near its hind legs. A small green collar is visible around its neck.
  - On the right is a larger, adult tabby cat with similar striped markings but more pronounced orange-brown tones. It’s curled up in a tight ball, head tucked down, appearing deeply asleep.

- **The Setting**:
  - The cats are resting on a soft, bright pink fleece-like blanket that covers most of the frame.
  - Behind them, the deep red fabric of the couch cushions is visible.
  - Two remote controls lie between and beside the cats — one white remote near the kitten’s head, and another white remote with colorful buttons positioned between the two cats.

- **Mood &amp; Composition**:
  - The image exudes warmth, comfort, and tranquility. The contrasting colors — pink blanket, red couch, and the cats’ natural tabby patterns — create a visually appealing scene.
  - The casual placement of the remotes suggests a lived-in, relaxed home environment where pets are welcome companions.

Overall, it’s a charming snapshot of feline relaxation and domestic peace.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This is a cozy, overhead photograph of two cats sleeping on a vibrant pink blanket draped over a red couch.

- **The Cats**:
  - On the left is a smaller cat, likely a kitten or young tabby, with classic brown and black stripes. It’s stretched out on its side, paws extended, with a fluffy tail curled near its hind legs. A small green collar is visible around its neck.
  - On the right is a larger, adult tabby cat with similar striped markings but more pronounced orange-brown tones. It’s curled up in a tight ball, head tucked down, appearing deeply asleep.

- **The Setting**:
  - The cats are resting on a soft, bright pink fleece-like blanket that covers most of the frame.
  - Behind them, the deep red fabric of the couch cushions is visible.
  - Two remote controls lie between and beside the cats — one white remote near the kitten’s head, and another white remote with colorful buttons positioned between the two cats.

- **Mood & Composition**:
  - The image exudes warmth, comfort, and tranquility. The contrasting colors — pink blanket, red couch, and the cats’ natural tabby patterns — create a visually appealing scene.
  - The casual placement of the remotes suggests a lived-in, relaxed home environment where pets are welcome companions.

Overall, it’s a charming snapshot of feline relaxation and domestic peace.
```

</details>

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
- *Model load time:* 9.17s
- *Generation time:* 9.61s
- *Total time:* 18.79s
- *Input validation time:* 0.00182
- *Prompt preparation time:* 0.00026
- *First-token latency:* 5.25
- *Cleanup time:* 0.257
- *Prompt tokens:* 317
- *Generation tokens:* 299
- *Total tokens:* 616
- *Prompt throughput (raw):* 60.4 tok/s
- *Generation throughput (raw):* 68.8 tok/s
- *Peak memory:* 71
- *Active memory:* 70
- *Cache memory:* 0.10
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.0137
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 131
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
This is a cozy, overhead photograph of two tabby cats sleeping peacefully on a vibrant pink couch or blanket.

- **The Cats**:
  - Both cats are classic brown tabbies with black stripes and swirls.
  - The cat on the left is stretched out diagonally, lying on its side with its head near the top left corner. Its tail curls slightly at the bottom. It appears to be wearing a small green collar.
  - The cat on the right is curled more compactly, facing downward toward the bottom right, with its head tucked in. Its body forms a gentle curve.
  - Both appear deeply asleep, exuding comfort and tranquility.

- **The Setting**:
  - They’re resting on a bright, fuchsia-pink fabric that looks soft — possibly a fleece blanket or the upholstery of a sofa.
  - Behind them, you can see the red or maroon backrest of the couch.

- **Objects**:
  - Two remote controls are visible:
    - One white remote lies horizontally near the top left, next to the first cat’s head.
    - Another white remote (with colorful buttons) is placed vertically between the two cats.

The overall mood of the image is warm, relaxed, and domestic — capturing a quiet moment of feline contentment. The contrast between the bold pink background and the natural tones of the cats makes for a visually striking and endearing scene.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This is a cozy, overhead photograph of two tabby cats sleeping peacefully on a vibrant pink couch or blanket.

- **The Cats**:
  - Both cats are classic brown tabbies with black stripes and swirls.
  - The cat on the left is stretched out diagonally, lying on its side with its head near the top left corner. Its tail curls slightly at the bottom. It appears to be wearing a small green collar.
  - The cat on the right is curled more compactly, facing downward toward the bottom right, with its head tucked in. Its body forms a gentle curve.
  - Both appear deeply asleep, exuding comfort and tranquility.

- **The Setting**:
  - They’re resting on a bright, fuchsia-pink fabric that looks soft — possibly a fleece blanket or the upholstery of a sofa.
  - Behind them, you can see the red or maroon backrest of the couch.

- **Objects**:
  - Two remote controls are visible:
    - One white remote lies horizontally near the top left, next to the first cat’s head.
    - Another white remote (with colorful buttons) is placed vertically between the two cats.

The overall mood of the image is warm, relaxed, and domestic — capturing a quiet moment of feline contentment. The contrast between the bold pink background and the natural tones of the cats makes for a visually striking and endearing scene.
```

</details>

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
- *Model load time:* 1.40s
- *Generation time:* 2.93s
- *Total time:* 4.34s
- *Input validation time:* 0.00166
- *Prompt preparation time:* 0.000241
- *First-token latency:* 0.225
- *Cleanup time:* 0.083
- *Prompt tokens:* 317
- *Generation tokens:* 265
- *Total tokens:* 582
- *Prompt throughput (raw):* 1,406 tok/s
- *Generation throughput (raw):* 100 tok/s
- *Peak memory:* 7.0
- *Active memory:* 6.7
- *Cache memory:* 0.11
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.0143
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 131
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
This is a cozy, heartwarming photograph of two tabby cats sleeping peacefully on a bright pink blanket or couch cover.

**Details:**

- **Cats:** Both are tabby cats with classic striped coats — one appears to be a younger kitten (on the left), and the other an adult cat (on the right). The kitten has a green collar and is curled up on its side, while the adult cat is lying on its back with paws stretched out, looking very relaxed.

- **Positioning:** They are lying next to each other, almost mirroring each other’s poses, which gives the image a sense of companionship and comfort.

- **Remote Controls:** Two white remote controls are placed between them — one near the kitten’s head and one near the adult cat’s head — suggesting they might be watching TV together before falling asleep.

- **Background:** The background is a vibrant, solid pink fabric — likely a sofa or cushion — which contrasts beautifully with the brown-and-black striped fur of the cats, making them stand out.

- **Mood:** The overall mood is tranquil, domestic, and affectionate — capturing a quiet moment of rest and bonding between two pets.

It’s a charming snapshot of everyday pet life, filled with warmth and simplicity.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This is a cozy, heartwarming photograph of two tabby cats sleeping peacefully on a bright pink blanket or couch cover.

**Details:**

- **Cats:** Both are tabby cats with classic striped coats — one appears to be a younger kitten (on the left), and the other an adult cat (on the right). The kitten has a green collar and is curled up on its side, while the adult cat is lying on its back with paws stretched out, looking very relaxed.
  
- **Positioning:** They are lying next to each other, almost mirroring each other’s poses, which gives the image a sense of companionship and comfort.

- **Remote Controls:** Two white remote controls are placed between them — one near the kitten’s head and one near the adult cat’s head — suggesting they might be watching TV together before falling asleep.

- **Background:** The background is a vibrant, solid pink fabric — likely a sofa or cushion — which contrasts beautifully with the brown-and-black striped fur of the cats, making them stand out.

- **Mood:** The overall mood is tranquil, domestic, and affectionate — capturing a quiet moment of rest and bonding between two pets.

It’s a charming snapshot of everyday pet life, filled with warmth and simplicity.
```

</details>

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
- *Model load time:* 3.08s
- *Generation time:* 16.29s
- *Total time:* 19.39s
- *Input validation time:* 0.00179
- *Prompt preparation time:* 0.0139
- *First-token latency:* 0.687
- *Cleanup time:* 0.13
- *Prompt tokens:* 317
- *Generation tokens:* 283
- *Total tokens:* 600
- *Prompt throughput (raw):* 461 tok/s
- *Generation throughput (raw):* 18.2 tok/s
- *Peak memory:* 30
- *Active memory:* 29
- *Cache memory:* 0.23
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.0148
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 131
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
This is a cozy, overhead photograph of two tabby cats lounging on a vibrant pink couch or blanket.

- **The Cats**: Both cats have classic tabby markings — stripes and swirls in shades of brown, black, and tan. The cat on the left is stretched out on its side, appearing to be asleep with its eyes closed. It has a green collar around its neck. The cat on the right is curled up more compactly, also seemingly asleep, with its head tucked down and paws tucked near its body.

- **The Setting**: They are resting on a bright, saturated pink fabric that covers what appears to be a sofa or cushioned surface. The color creates a strong contrast with the earthy tones of the cats’ fur.

- **Objects**: Two remote controls are visible on the pink surface — one near the head of the left cat (white with blue buttons) and another between the two cats (white with colorful buttons). Their presence suggests this is a living room setting where someone might have been watching TV before the cats claimed the space.

- **Mood**: The image conveys a sense of peace, comfort, and domestic tranquility. The cats look completely relaxed and content, embodying the lazy, sleepy vibe of a quiet afternoon at home.

Overall, it’s a charming snapshot of feline relaxation in a colorful, everyday environment.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This is a cozy, overhead photograph of two tabby cats lounging on a vibrant pink couch or blanket.

- **The Cats**: Both cats have classic tabby markings — stripes and swirls in shades of brown, black, and tan. The cat on the left is stretched out on its side, appearing to be asleep with its eyes closed. It has a green collar around its neck. The cat on the right is curled up more compactly, also seemingly asleep, with its head tucked down and paws tucked near its body.

- **The Setting**: They are resting on a bright, saturated pink fabric that covers what appears to be a sofa or cushioned surface. The color creates a strong contrast with the earthy tones of the cats’ fur.

- **Objects**: Two remote controls are visible on the pink surface — one near the head of the left cat (white with blue buttons) and another between the two cats (white with colorful buttons). Their presence suggests this is a living room setting where someone might have been watching TV before the cats claimed the space.

- **Mood**: The image conveys a sense of peace, comfort, and domestic tranquility. The cats look completely relaxed and content, embodying the lazy, sleepy vibe of a quiet afternoon at home.

Overall, it’s a charming snapshot of feline relaxation in a colorful, everyday environment.
```

</details>

</details>

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### mlx-community/SmolVLM-Instruct-bf16

<details>
<summary>Complete evidence: mlx-community/SmolVLM-Instruct-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.60s
- *Generation time:* 0.98s
- *Total time:* 1.58s
- *Input validation time:* 0.00167
- *Prompt preparation time:* 0.00115
- *First-token latency:* 0.405
- *Cleanup time:* 0.077
- *Prompt tokens:* 1,194
- *Generation tokens:* 63
- *Total tokens:* 1,257
- *Prompt throughput (raw):* 2,945 tok/s
- *Generation throughput (raw):* 127 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.30
- *Model-load active memory:* 4.51
- *Post-cleanup active memory:* 0.0149
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 72
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
 Two cats are lying on a pink blanket on a couch. The cat on the left is curled up with its head on the cat on the right's back. The cat on the right is stretched out with its head on the cat on the left's back. There are two remote controls on the blanket.
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Two cats are lying on a pink blanket on a couch. The cat on the left is curled up with its head on the cat on the right's back. The cat on the right is stretched out with its head on the cat on the left's back. There are two remote controls on the blanket.
```

</details>

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
- *Model load time:* 0.60s
- *Generation time:* 1.43s
- *Total time:* 2.04s
- *Input validation time:* 0.0016
- *Prompt preparation time:* 0.000945
- *First-token latency:* 0.278
- *Cleanup time:* 0.0728
- *Prompt tokens:* 95
- *Generation tokens:* 143
- *Total tokens:* 238
- *Prompt throughput (raw):* 342 tok/s
- *Generation throughput (raw):* 132 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.09
- *Model-load active memory:* 4.51
- *Post-cleanup active memory:* 0.015
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 72
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
 The image depicts a scene featuring a cat and a remote control. The cat, which appears to be a tabby with a mix of black, white, and gray fur, is lying on a red couch. The cat's body is mostly covered by the couch, with its head resting on the armrest. The remote control, which is white with a blue and red button, is placed on the armrest of the couch, close to the cat's head. The remote control is positioned horizontally, with the buttons facing upwards. The couch itself is red, and it appears to be a standard piece of furniture with a simple design. The overall scene suggests a relaxed and comfortable setting, possibly a home environment.
</pre>

<details>
<summary>Exact raw output</summary>

```text
 The image depicts a scene featuring a cat and a remote control. The cat, which appears to be a tabby with a mix of black, white, and gray fur, is lying on a red couch. The cat's body is mostly covered by the couch, with its head resting on the armrest. The remote control, which is white with a blue and red button, is placed on the armrest of the couch, close to the cat's head. The remote control is positioned horizontally, with the buttons facing upwards. The couch itself is red, and it appears to be a standard piece of furniture with a simple design. The overall scene suggests a relaxed and comfortable setting, possibly a home environment.
```

</details>

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
- *Model load time:* 6.57s
- *Generation time:* 7.91s
- *Total time:* 14.50s
- *Input validation time:* 0.00173
- *Prompt preparation time:* 0.0133
- *First-token latency:* 3.91
- *Cleanup time:* 0.228
- *Prompt tokens:* 188
- *Generation tokens:* 172
- *Total tokens:* 360
- *Prompt throughput (raw):* 48.1 tok/s
- *Generation throughput (raw):* 43.7 tok/s
- *Peak memory:* 65
- *Active memory:* 63
- *Cache memory:* 0.10
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.0153
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 118
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
The image shows two cats sleeping on a pink blanket or sheet. The cat on the left is a tabby cat with a green collar, lying on its side with its head resting on a folded part of the blanket. The cat on the right is a larger tabby cat, also lying on its side, facing the same direction as the smaller cat. Both cats appear to be in a relaxed, sleeping state. There are two remote controls placed near the cats, one on each side of them. The remote on the left is a standard TV remote with multiple buttons, and the one on the right is a simpler, white remote with fewer buttons. The background includes a red cushion or sofa behind the cats, adding a contrasting color to the scene. The overall atmosphere is calm and cozy, with the cats seemingly comfortable and at ease in their resting positions.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two cats sleeping on a pink blanket or sheet. The cat on the left is a tabby cat with a green collar, lying on its side with its head resting on a folded part of the blanket. The cat on the right is a larger tabby cat, also lying on its side, facing the same direction as the smaller cat. Both cats appear to be in a relaxed, sleeping state. There are two remote controls placed near the cats, one on each side of them. The remote on the left is a standard TV remote with multiple buttons, and the one on the right is a simpler, white remote with fewer buttons. The background includes a red cushion or sofa behind the cats, adding a contrasting color to the scene. The overall atmosphere is calm and cozy, with the cats seemingly comfortable and at ease in their resting positions.
```

</details>

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
- *Model load time:* 1.15s
- *Generation time:* 1.89s
- *Total time:* 3.05s
- *Input validation time:* 0.00164
- *Prompt preparation time:* 0.000257
- *First-token latency:* 0.239
- *Cleanup time:* 0.0928
- *Prompt tokens:* 415
- *Generation tokens:* 104
- *Total tokens:* 519
- *Prompt throughput (raw):* 1,737 tok/s
- *Generation throughput (raw):* 66.3 tok/s
- *Peak memory:* 10
- *Active memory:* 9.5
- *Cache memory:* 0.06
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0156
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 170
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
The image shows two cats lying on a pink couch. Both cats have a tabby pattern with brown, black, and white fur. The cat on the left is curled up with its head resting on the back of the couch, while the cat on the right is stretched out on its side. Between them are two remote controls, one near each cat. The couch appears to be a vibrant pink color, and the overall scene conveys a cozy and relaxed atmosphere. The cats seem to be resting or napping.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two cats lying on a pink couch. Both cats have a tabby pattern with brown, black, and white fur. The cat on the left is curled up with its head resting on the back of the couch, while the cat on the right is stretched out on its side. Between them are two remote controls, one near each cat. The couch appears to be a vibrant pink color, and the overall scene conveys a cozy and relaxed atmosphere. The cats seem to be resting or napping.
```

</details>

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
- *Model load time:* 2.32s
- *Generation time:* 10.89s
- *Total time:* 13.23s
- *Input validation time:* 0.00164
- *Prompt preparation time:* 0.00272
- *First-token latency:* 0.704
- *Cleanup time:* 0.119
- *Prompt tokens:* 273
- *Generation tokens:* 303
- *Total tokens:* 576
- *Prompt throughput (raw):* 388 tok/s
- *Generation throughput (raw):* 29.8 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.41
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0161
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 95
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
Here's a description of the image:

**Overall Impression:**

The image is a charming and humorous snapshot of two tabby cats lounging on a bright pink surface, likely a couch cushion or blanket. They are positioned as if they are "controlling" the TV with remote controls.

**Detailed Description:**

*   **Cats:** There are two tabby cats. Both have striped fur patterns, but with slightly different shades and markings. The cat on the left has darker, more defined stripes and a more silver tone. The cat on the right has warmer, more golden tones in its stripes. Both cats are lying on their backs, appearing very relaxed and sleepy.
*   **Remote Controls:** Each cat has a remote control positioned near them, as if they are holding or using it. This adds to the playful and humorous nature of the image.
*   **Surface:** The cats are lying on a vibrant pink surface. It looks soft and comfortable, likely a couch cushion or blanket.
*   **Composition:** The cats are the clear focal point of the image. The bright pink background provides a strong contrast, making them stand out. The positioning of the remote controls adds to the narrative of the image.
*   **Mood:** The image evokes a feeling of warmth, comfort, and playfulness. It's a lighthearted and endearing scene.

In essence, it's a cute and funny picture of cats enjoying a lazy day, seemingly in control of their entertainment!
</pre>

<details>
<summary>Exact raw output</summary>

```text
Here's a description of the image:

**Overall Impression:**

The image is a charming and humorous snapshot of two tabby cats lounging on a bright pink surface, likely a couch cushion or blanket. They are positioned as if they are "controlling" the TV with remote controls.

**Detailed Description:**

*   **Cats:** There are two tabby cats. Both have striped fur patterns, but with slightly different shades and markings. The cat on the left has darker, more defined stripes and a more silver tone. The cat on the right has warmer, more golden tones in its stripes. Both cats are lying on their backs, appearing very relaxed and sleepy.
*   **Remote Controls:** Each cat has a remote control positioned near them, as if they are holding or using it. This adds to the playful and humorous nature of the image.
*   **Surface:** The cats are lying on a vibrant pink surface. It looks soft and comfortable, likely a couch cushion or blanket.
*   **Composition:** The cats are the clear focal point of the image. The bright pink background provides a strong contrast, making them stand out. The positioning of the remote controls adds to the narrative of the image.
*   **Mood:** The image evokes a feeling of warmth, comfort, and playfulness. It's a lighthearted and endearing scene.

In essence, it's a cute and funny picture of cats enjoying a lazy day, seemingly in control of their entertainment!
```

</details>

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
- *Model load time:* 3.27s
- *Generation time:* 18.26s
- *Total time:* 21.54s
- *Input validation time:* 0.00204
- *Prompt preparation time:* 0.000204
- *First-token latency:* 0.838
- *Cleanup time:* 0.149
- *Prompt tokens:* 273
- *Generation tokens:* 289
- *Total tokens:* 562
- *Prompt throughput (raw):* 326 tok/s
- *Generation throughput (raw):* 16.6 tok/s
- *Peak memory:* 32
- *Active memory:* 30
- *Cache memory:* 0.41
- *Model-load active memory:* 29.7
- *Post-cleanup active memory:* 0.0166
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 95
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
Here's a description of the image:

**Overall Impression:**

The image is a charming and playful shot of two tabby kittens lounging on a bright pink surface, likely a couch cushion or blanket. They are positioned as if they are "watching TV," with remote controls placed near them.

**Detailed Description:**

*   **Kittens:** There are two kittens, both with classic tabby markings (stripes and swirls). One kitten has a more grey/brown tabby pattern, while the other is a warmer, more orange-toned tabby. Both kittens are lying on their backs with their paws curled up, appearing very relaxed and sleepy. They both have collars on.
*   **Positioning:** The kittens are positioned side-by-side, with each one having a remote control nearby. It looks like they are mimicking people watching television.
*   **Remote Controls:** Two remote controls are visible, one grey and one beige, placed near each kitten.
*   **Surface:** The kittens are lying on a vibrant pink surface. The texture appears soft, suggesting a couch cushion or blanket.
*   **Lighting:** The lighting is fairly even, and the colors are bright and clear.
*   **Overall Tone:** The image is lighthearted, cute, and humorous. It evokes a feeling of warmth and relaxation.

It's a very endearing picture that captures the playful nature of kittens!
</pre>

<details>
<summary>Exact raw output</summary>

```text
Here's a description of the image:

**Overall Impression:**

The image is a charming and playful shot of two tabby kittens lounging on a bright pink surface, likely a couch cushion or blanket. They are positioned as if they are "watching TV," with remote controls placed near them.

**Detailed Description:**

*   **Kittens:** There are two kittens, both with classic tabby markings (stripes and swirls). One kitten has a more grey/brown tabby pattern, while the other is a warmer, more orange-toned tabby. Both kittens are lying on their backs with their paws curled up, appearing very relaxed and sleepy. They both have collars on.
*   **Positioning:** The kittens are positioned side-by-side, with each one having a remote control nearby. It looks like they are mimicking people watching television.
*   **Remote Controls:** Two remote controls are visible, one grey and one beige, placed near each kitten.
*   **Surface:** The kittens are lying on a vibrant pink surface. The texture appears soft, suggesting a couch cushion or blanket.
*   **Lighting:** The lighting is fairly even, and the colors are bright and clear.
*   **Overall Tone:** The image is lighthearted, cute, and humorous. It evokes a feeling of warmth and relaxation.

It's a very endearing picture that captures the playful nature of kittens!
```

</details>

</details>

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### mlx-community/gemma-3n-E4B-it-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E4B-it-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 2.43s
- *Generation time:* 2.53s
- *Total time:* 4.97s
- *Input validation time:* 0.00175
- *Prompt preparation time:* 0.00292
- *First-token latency:* 0.325
- *Cleanup time:* 0.117
- *Prompt tokens:* 272
- *Generation tokens:* 106
- *Total tokens:* 378
- *Prompt throughput (raw):* 837 tok/s
- *Generation throughput (raw):* 48.4 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.04
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0177
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 97
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
Two tabby cats are peacefully napping on a bright pink couch. The cat on the left is lying on its back with its paws outstretched, while the cat on the right is curled up on its side, also seemingly asleep.

Between the two cats are two television remotes, one white and one blue. The couch appears to be made of a soft, plush material. The lighting in the image is soft and natural, suggesting it might be daytime. The overall mood of the image is one of tranquility and relaxation.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Two tabby cats are peacefully napping on a bright pink couch. The cat on the left is lying on its back with its paws outstretched, while the cat on the right is curled up on its side, also seemingly asleep. 

Between the two cats are two television remotes, one white and one blue. The couch appears to be made of a soft, plush material. The lighting in the image is soft and natural, suggesting it might be daytime. The overall mood of the image is one of tranquility and relaxation.
```

</details>

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
- *Model load time:* 2.40s
- *Generation time:* 1.55s
- *Total time:* 3.98s
- *Input validation time:* 0.00163
- *Prompt preparation time:* 0.021
- *First-token latency:* 0.388
- *Cleanup time:* 0.119
- *Prompt tokens:* 284
- *Generation tokens:* 148
- *Total tokens:* 432
- *Prompt throughput (raw):* 732 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.13
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0183
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 94
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
A high-angle, medium shot shows two tabby cats lying on a pink blanket on a red couch. The cat on the left is lying on its side, facing right, with its head towards the top right corner. It has dark brown and black stripes on its body and a dark tail. Its front paws are tucked under its body. The cat on the right is lying on its side, facing left, with its head towards the bottom left corner. It has light brown and black stripes on its body and a dark tail. Its front paws are stretched out towards the bottom right corner. A silver remote control is lying on the pink blanket between the two cats. The background is a red couch. The lighting is bright and even.
</pre>

<details>
<summary>Exact raw output</summary>

```text
A high-angle, medium shot shows two tabby cats lying on a pink blanket on a red couch. The cat on the left is lying on its side, facing right, with its head towards the top right corner. It has dark brown and black stripes on its body and a dark tail. Its front paws are tucked under its body. The cat on the right is lying on its side, facing left, with its head towards the bottom left corner. It has light brown and black stripes on its body and a dark tail. Its front paws are stretched out towards the bottom right corner. A silver remote control is lying on the pink blanket between the two cats. The background is a red couch. The lighting is bright and even.
```

</details>

</details>

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### mlx-community/gemma-4-31b-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-4-31b-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 6.96s
- *Generation time:* 5.57s
- *Total time:* 12.54s
- *Input validation time:* 0.00164
- *Prompt preparation time:* 0.000148
- *First-token latency:* 3.65
- *Cleanup time:* 0.237
- *Prompt tokens:* 272
- *Generation tokens:* 15
- *Total tokens:* 287
- *Prompt throughput (raw):* 74.5 tok/s
- *Generation throughput (raw):* 7.92 tok/s
- *Peak memory:* 63
- *Active memory:* 63
- *Cache memory:* 0.49
- *Model-load active memory:* 62.6
- *Post-cleanup active memory:* 0.0188
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 29
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

*Readable output:*

<pre class="model-output-readable">
 in a single sentence.Both cats are sleeping on a pink blanket.
</pre>

<details>
<summary>Exact raw output</summary>

```text
 in a single sentence.Both cats are sleeping on a pink blanket.
```

</details>

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
- *Model load time:* 2.54s
- *Generation time:* 4.33s
- *Total time:* 6.88s
- *Input validation time:* 0.00173
- *Prompt preparation time:* 0.000279
- *First-token latency:* 0.712
- *Cleanup time:* 0.131
- *Prompt tokens:* 284
- *Generation tokens:* 98
- *Total tokens:* 382
- *Prompt throughput (raw):* 399 tok/s
- *Generation throughput (raw):* 27.2 tok/s
- *Peak memory:* 19
- *Active memory:* 18
- *Cache memory:* 0.50
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0193
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 94
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
A high-angle, medium shot shows two brown tabby cats lying on a bright pink blanket on a red couch. The cat on the left is lying on its side, facing right, with its head resting on a blue remote control. The cat on the right is lying on its back, facing left, with its head resting on a white remote control. Both cats have their eyes closed and appear to be sleeping. The background is a red couch. The lighting is bright and even.
</pre>

<details>
<summary>Exact raw output</summary>

```text
A high-angle, medium shot shows two brown tabby cats lying on a bright pink blanket on a red couch. The cat on the left is lying on its side, facing right, with its head resting on a blue remote control. The cat on the right is lying on its back, facing left, with its head resting on a white remote control. Both cats have their eyes closed and appear to be sleeping. The background is a red couch. The lighting is bright and even.
```

</details>

</details>

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### mlx-community/llava-v1.6-mistral-7b-8bit

<details>
<summary>Complete evidence: mlx-community/llava-v1.6-mistral-7b-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.92s
- *Generation time:* 3.98s
- *Total time:* 4.90s
- *Input validation time:* 0.00163
- *Prompt preparation time:* 0.00157
- *First-token latency:* 2.6
- *Cleanup time:* 0.0939
- *Prompt tokens:* 2,354
- *Generation tokens:* 82
- *Total tokens:* 2,436
- *Prompt throughput (raw):* 906 tok/s
- *Generation throughput (raw):* 62.5 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.83
- *Model-load active memory:* 8.06
- *Post-cleanup active memory:* 0.0194
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 42
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
The image shows two cats lying on a pink blanket. The cat on the left is a kitten with a black and white striped coat, while the cat on the right is a tabby cat with a brown and black coat. Both cats are sleeping or resting. In the background, there is a remote control on the blanket. The cats are on a couch, and the blanket is pink.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two cats lying on a pink blanket. The cat on the left is a kitten with a black and white striped coat, while the cat on the right is a tabby cat with a brown and black coat. Both cats are sleeping or resting. In the background, there is a remote control on the blanket. The cats are on a couch, and the blanket is pink. 
```

</details>

</details>

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### mlx-community/nanoLLaVA-1.5-4bit

<details>
<summary>Complete evidence: mlx-community/nanoLLaVA-1.5-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.55s
- *Generation time:* 0.41s
- *Total time:* 0.96s
- *Input validation time:* 0.00174
- *Prompt preparation time:* 0.00143
- *First-token latency:* 0.0674
- *Cleanup time:* 0.079
- *Prompt tokens:* 20
- *Generation tokens:* 123
- *Total tokens:* 143
- *Prompt throughput (raw):* 297 tok/s
- *Generation throughput (raw):* 378 tok/s
- *Peak memory:* 1.8
- *Active memory:* 0.62
- *Cache memory:* 0.24
- *Model-load active memory:* 0.618
- *Post-cleanup active memory:* 0.02
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 126
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
The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats are wearing striped-patterned clothing, and they are positioned in such a way that they appear to be sleeping. The cats are facing the camera, and their eyes are wide open, indicating they are alert and aware of their surroundings. The background of the image is a solid pink color, providing a contrast to the cats and the fabric surface. There are no texts or other objects in the image. The style of the image is a straightforward and unembellished photograph, capturing the cats in a relaxed state.
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats are wearing striped-patterned clothing, and they are positioned in such a way that they appear to be sleeping. The cats are facing the camera, and their eyes are wide open, indicating they are alert and aware of their surroundings. The background of the image is a solid pink color, providing a contrast to the cats and the fabric surface. There are no texts or other objects in the image. The style of the image is a straightforward and unembellished photograph, capturing the cats in a relaxed state.
```

</details>

</details>

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### mlx-community/paligemma2-10b-ft-docci-448-6bit

<details>
<summary>Complete evidence: mlx-community/paligemma2-10b-ft-docci-448-6bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 1.58s
- *Generation time:* 6.49s
- *Total time:* 8.08s
- *Input validation time:* 0.00155
- *Prompt preparation time:* 0.000178
- *First-token latency:* 0.782
- *Cleanup time:* 0.127
- *Prompt tokens:* 1,029
- *Generation tokens:* 180
- *Total tokens:* 1,209
- *Prompt throughput (raw):* 1,316 tok/s
- *Generation throughput (raw):* 32.6 tok/s
- *Peak memory:* 11
- *Active memory:* 7.9
- *Cache memory:* 1.5
- *Model-load active memory:* 7.88
- *Post-cleanup active memory:* 0.021
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 26
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
Two cats are laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes. The cat on the right is a brown tabby cat with black stripes. Both cats are laying on their sides. The cat on the left is facing the right side of the image. The cat on the right is facing the left side of the image. The cat on the left has its head down, and its body is stretched out. Its head is facing the top left corner of the image. Its tail is hanging off the edge of the blanket. The cat on the right has its head down, and its body is stretched out. Its head is facing the bottom left corner of the image. Its tail is hanging off the edge of the blanket. There is a gray remote control on the left side of the image and a gray remote control on the right side of the image.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Two cats are laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes. The cat on the right is a brown tabby cat with black stripes. Both cats are laying on their sides. The cat on the left is facing the right side of the image. The cat on the right is facing the left side of the image. The cat on the left has its head down, and its body is stretched out. Its head is facing the top left corner of the image. Its tail is hanging off the edge of the blanket. The cat on the right has its head down, and its body is stretched out. Its head is facing the bottom left corner of the image. Its tail is hanging off the edge of the blanket. There is a gray remote control on the left side of the image and a gray remote control on the right side of the image.
```

</details>

</details>

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### mlx-community/paligemma2-10b-ft-docci-448-bf16

<details>
<summary>Complete evidence: mlx-community/paligemma2-10b-ft-docci-448-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 2.75s
- *Generation time:* 30.54s
- *Total time:* 33.30s
- *Input validation time:* 0.00171
- *Prompt preparation time:* 0.00013
- *First-token latency:* 1.12
- *Cleanup time:* 0.131
- *Prompt tokens:* 1,029
- *Generation tokens:* 159
- *Total tokens:* 1,188
- *Prompt throughput (raw):* 923 tok/s
- *Generation throughput (raw):* 5.41 tok/s
- *Peak memory:* 26
- *Active memory:* 19
- *Cache memory:* 6.2
- *Model-load active memory:* 19.3
- *Post-cleanup active memory:* 0.0221
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 26
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
A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and a black tail. It is laying on its side with its head on the pink blanket and its body facing the left side of the image. Its front legs are stretched out in front of it, and its back legs are curled up. The cat on the right is a brown tabby cat with black stripes. It is laying on its side with its head on the pink blanket and its body facing the right side of the image. Its front legs are stretched out in front of it, and its back legs are curled up. There is a gray remote control on the left side of the image and a gray remote control on the right side of the image.
</pre>

<details>
<summary>Exact raw output</summary>

```text
A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and a black tail. It is laying on its side with its head on the pink blanket and its body facing the left side of the image. Its front legs are stretched out in front of it, and its back legs are curled up. The cat on the right is a brown tabby cat with black stripes. It is laying on its side with its head on the pink blanket and its body facing the right side of the image. Its front legs are stretched out in front of it, and its back legs are curled up. There is a gray remote control on the left side of the image and a gray remote control on the right side of the image.
```

</details>

</details>

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### mlx-community/paligemma2-3b-ft-docci-448-bf16

<details>
<summary>Complete evidence: mlx-community/paligemma2-3b-ft-docci-448-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 1.42s
- *Generation time:* 6.78s
- *Total time:* 8.20s
- *Input validation time:* 0.00164
- *Prompt preparation time:* 0.00013
- *First-token latency:* 0.374
- *Cleanup time:* 0.0975
- *Prompt tokens:* 1,029
- *Generation tokens:* 126
- *Total tokens:* 1,155
- *Prompt throughput (raw):* 2,753 tok/s
- *Generation throughput (raw):* 19.7 tok/s
- *Peak memory:* 10
- *Active memory:* 6.1
- *Cache memory:* 3.6
- *Model-load active memory:* 6.09
- *Post-cleanup active memory:* 0.0231
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 26
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
A top-down view of two cats sleeping on a pink blanket. The cat on the left is a gray and black color and is laying on its side with its head facing the right side of the image. Its front paws are stretched out in front of it. The cat's tail is sticking out to the left of the blanket. The cat on the right is lying on its side with its head facing the left side of the image. Its front paws are stretched out in front of it. Its tail is sticking out to the left of the blanket. There is a white and gray remote control between the two cats.
</pre>

<details>
<summary>Exact raw output</summary>

```text
A top-down view of two cats sleeping on a pink blanket. The cat on the left is a gray and black color and is laying on its side with its head facing the right side of the image. Its front paws are stretched out in front of it. The cat's tail is sticking out to the left of the blanket. The cat on the right is lying on its side with its head facing the left side of the image. Its front paws are stretched out in front of it. Its tail is sticking out to the left of the blanket. There is a white and gray remote control between the two cats.
```

</details>

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
- *Model load time:* 1.73s
- *Generation time:* 5.73s
- *Total time:* 7.47s
- *Input validation time:* 0.00163
- *Prompt preparation time:* 0.00223
- *First-token latency:* 0.77
- *Cleanup time:* 0.0981
- *Prompt tokens:* 1,237
- *Generation tokens:* 199
- *Total tokens:* 1,436
- *Prompt throughput (raw):* 1,606 tok/s
- *Generation throughput (raw):* 40.2 tok/s
- *Peak memory:* 15
- *Active memory:* 14
- *Cache memory:* 0.27
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0244
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 40
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
In the tranquil setting of this image, two feline companions, one adult and one kitten, are enjoying a peaceful slumber on a vibrant red couch. The adult cat, with its brown and black stripes, is lying on its side, its head comfortably resting on the armrest of the couch. Its body is relaxed, embodying the essence of serenity.

Beside it, the kitten, mirroring the adult's brown and black stripes, is curled up in a perfect circle. Its head is gently resting on the seat of the couch, adding to the overall calmness of the scene.

The couch they're on is a striking red, providing a warm contrast to the cats' brown and black fur. On the armrest of the couch, there are two remote controls, perhaps indicating a recent human presence. The precise location of these objects and their relative positions to each other and the cats add depth to the image, creating a snapshot of a quiet moment in time.
</pre>

<details>
<summary>Exact raw output</summary>

```text
In the tranquil setting of this image, two feline companions, one adult and one kitten, are enjoying a peaceful slumber on a vibrant red couch. The adult cat, with its brown and black stripes, is lying on its side, its head comfortably resting on the armrest of the couch. Its body is relaxed, embodying the essence of serenity.

Beside it, the kitten, mirroring the adult's brown and black stripes, is curled up in a perfect circle. Its head is gently resting on the seat of the couch, adding to the overall calmness of the scene.

The couch they're on is a striking red, providing a warm contrast to the cats' brown and black fur. On the armrest of the couch, there are two remote controls, perhaps indicating a recent human presence. The precise location of these objects and their relative positions to each other and the cats add depth to the image, creating a snapshot of a quiet moment in time.
```

</details>

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
- *Model load time:* 2.60s
- *Generation time:* 11.00s
- *Total time:* 13.60s
- *Input validation time:* 0.00183
- *Prompt preparation time:* 0.000195
- *First-token latency:* 0.785
- *Cleanup time:* 0.11
- *Prompt tokens:* 1,237
- *Generation tokens:* 204
- *Total tokens:* 1,441
- *Prompt throughput (raw):* 1,577 tok/s
- *Generation throughput (raw):* 20.0 tok/s
- *Peak memory:* 27
- *Active memory:* 25
- *Cache memory:* 0.27
- *Model-load active memory:* 25.4
- *Post-cleanup active memory:* 0.0247
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 40
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
In the tranquil setting of this image, two feline companions, one a tabby cat and the other a Bengal cat, are enjoying a peaceful slumber on a vibrant red couch. The tabby cat, with its distinctive striped coat, is sprawled out on its side, its head comfortably resting on the armrest of the couch. Its body is relaxed, embodying the essence of serenity.

Beside it, the Bengal cat is curled up in a perfect circle, its head nestled on its paws. The intricate patterns on its coat blend harmoniously with the red of the couch, creating a picture of tranquility.

Two remote controls lie on the couch, silent witnesses to the cats' slumber. One is near the tabby cat, while the other is closer to the Bengal cat. Their presence adds a touch of everyday life to this serene scene.

The image captures a moment of pure peace and comfort, as these two cats enjoy their rest in a warm and inviting environment.
</pre>

<details>
<summary>Exact raw output</summary>

```text
In the tranquil setting of this image, two feline companions, one a tabby cat and the other a Bengal cat, are enjoying a peaceful slumber on a vibrant red couch. The tabby cat, with its distinctive striped coat, is sprawled out on its side, its head comfortably resting on the armrest of the couch. Its body is relaxed, embodying the essence of serenity.

Beside it, the Bengal cat is curled up in a perfect circle, its head nestled on its paws. The intricate patterns on its coat blend harmoniously with the red of the couch, creating a picture of tranquility.

Two remote controls lie on the couch, silent witnesses to the cats' slumber. One is near the tabby cat, while the other is closer to the Bengal cat. Their presence adds a touch of everyday life to this serene scene.

The image captures a moment of pure peace and comfort, as these two cats enjoy their rest in a warm and inviting environment.
```

</details>

</details>

---

<a id="model-qnguyen3-nanollava"></a>

### qnguyen3/nanoLLaVA

<details>
<summary>Complete evidence: qnguyen3/nanoLLaVA</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 0.50s
- *Generation time:* 0.50s
- *Total time:* 1.00s
- *Input validation time:* 0.00185
- *Prompt preparation time:* 0.000565
- *First-token latency:* 0.0895
- *Cleanup time:* 0.0891
- *Prompt tokens:* 20
- *Generation tokens:* 44
- *Total tokens:* 64
- *Prompt throughput (raw):* 223 tok/s
- *Generation throughput (raw):* 116 tok/s
- *Peak memory:* 4.2
- *Active memory:* 2.1
- *Cache memory:* 1.6
- *Model-load active memory:* 2.13
- *Post-cleanup active memory:* 0.0253
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 126
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
This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose. The cats are lying on a pink blanket.
</pre>

<details>
<summary>Exact raw output</summary>

```text
This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose. The cats are lying on a pink blanket.
```

</details>

</details>

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### jqlive/Kimi-VL-A3B-Thinking-2506-6bit

<details>
<summary>Complete evidence: jqlive/Kimi-VL-A3B-Thinking-2506-6bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Conversation-role control tokens remain visible
- *Model load time:* 1.97s
- *Generation time:* 3.25s
- *Total time:* 5.22s
- *Input validation time:* 0.00182
- *Prompt preparation time:* 0.00182
- *First-token latency:* 0.288
- *Cleanup time:* 0.0928
- *Prompt tokens:* 397
- *Generation tokens:* 221
- *Total tokens:* 618
- *Prompt throughput (raw):* 1,380 tok/s
- *Generation throughput (raw):* 77.6 tok/s
- *Peak memory:* 16
- *Active memory:* 14
- *Cache memory:* 0.70
- *Model-load active memory:* 13.8
- *Post-cleanup active memory:* 0.00121
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 227
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
◁think▷So, let's analyze the image. The image shows two tabby cats lying on a pink fabric surface. There are two remote controls, one near each cat. The cats have a mix of brown, black, and white fur. The lighting is high-contrast, giving a stylized, almost posterized look. The scene is vibrant with the pink background. The cats appear relaxed, possibly sleeping or resting. The remotes are light blue with buttons, typical of TV or other electronic remotes. The overall style is artistic with strong color saturation.◁/think▷The image depicts two tabby cats resting on a vibrant pink fabric surface. One cat is positioned on the left, with a light blue remote control nearby, while the other cat is on the right, with a similar remote. Both cats have a mix of brown, black, and white fur patterns. The image has a stylized, high-contrast, posterized aesthetic, emphasizing vivid colors and sharp details. The cats appear relaxed, likely sleeping or resting, against the bright pink background.&lt;|im_assistant|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷So, let's analyze the image. The image shows two tabby cats lying on a pink fabric surface. There are two remote controls, one near each cat. The cats have a mix of brown, black, and white fur. The lighting is high-contrast, giving a stylized, almost posterized look. The scene is vibrant with the pink background. The cats appear relaxed, possibly sleeping or resting. The remotes are light blue with buttons, typical of TV or other electronic remotes. The overall style is artistic with strong color saturation.◁/think▷The image depicts two tabby cats resting on a vibrant pink fabric surface. One cat is positioned on the left, with a light blue remote control nearby, while the other cat is on the right, with a similar remote. Both cats have a mix of brown, black, and white fur patterns. The image has a stylized, high-contrast, posterized aesthetic, emphasizing vivid colors and sharp details. The cats appear relaxed, likely sleeping or resting, against the bright pink background.<|im_assistant|>
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
- *Model load time:* 1.90s
- *Generation time:* 12.10s
- *Total time:* 14.00s
- *Input validation time:* 0.00166
- *Prompt preparation time:* 0.00115
- *First-token latency:* 1.04
- *Cleanup time:* 0.0996
- *Prompt tokens:* 2,325
- *Generation tokens:* 354
- *Total tokens:* 2,679
- *Prompt throughput (raw):* 2,225 tok/s
- *Generation throughput (raw):* 32.2 tok/s
- *Peak memory:* 19
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.00442
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 77
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
 The image depicts two cats lying on a pink fabric, possibly a blanket or a piece of clothing, which is draped over a piece of furniture, likely a couch or a chair. The cats are positioned side by side, and they appear to be in a relaxed state, possibly sleeping or resting. The fabric is a solid pink color, and it is smooth and soft-looking, suggesting it is made of a comfortable material like cotton or fleece.

The cats are both of a similar size and have a similar color pattern. They have a mix of brown and black stripes, with the brown stripes being more prominent. Their fur appears to be short and well-groomed, indicating that they are well-cared for. The cats' eyes are closed, and their ears are slightly tilted back, which is a common posture for cats when they are relaxed or sleeping.

In the background, there are two remote controls, one on each side of the cats. The remotes are white with black buttons, and they are placed on the pink fabric. The remotes are not in use, and they are simply resting on the fabric.

The overall scene is calm and serene, with the cats and remotes creating a cozy and comfortable setting. The pink fabric adds a soft and warm touch to the image, making it feel inviting and relaxing. The cats' presence adds a sense of companionship and warmth to the scene, making it a pleasant and homely image.

In summary, the image features two cats lying on a pink fabric, with two remotes in the background, creating a cozy and relaxed atmosphere. The cats are well-groomed and appear to be sleeping or resting, while the pink fabric and remotes add to the comfort and homeliness of the scene.&lt;end_of_utterance&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
 The image depicts two cats lying on a pink fabric, possibly a blanket or a piece of clothing, which is draped over a piece of furniture, likely a couch or a chair. The cats are positioned side by side, and they appear to be in a relaxed state, possibly sleeping or resting. The fabric is a solid pink color, and it is smooth and soft-looking, suggesting it is made of a comfortable material like cotton or fleece.

The cats are both of a similar size and have a similar color pattern. They have a mix of brown and black stripes, with the brown stripes being more prominent. Their fur appears to be short and well-groomed, indicating that they are well-cared for. The cats' eyes are closed, and their ears are slightly tilted back, which is a common posture for cats when they are relaxed or sleeping.

In the background, there are two remote controls, one on each side of the cats. The remotes are white with black buttons, and they are placed on the pink fabric. The remotes are not in use, and they are simply resting on the fabric.

The overall scene is calm and serene, with the cats and remotes creating a cozy and comfortable setting. The pink fabric adds a soft and warm touch to the image, making it feel inviting and relaxing. The cats' presence adds a sense of companionship and warmth to the scene, making it a pleasant and homely image.

In summary, the image features two cats lying on a pink fabric, with two remotes in the background, creating a cozy and relaxed atmosphere. The cats are well-groomed and appear to be sleeping or resting, while the pink fabric and remotes add to the comfort and homeliness of the scene.<end_of_utterance>
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
- *Model load time:* 3.24s
- *Generation time:* 52.33s
- *Total time:* 55.58s
- *Input validation time:* 0.00179
- *Prompt preparation time:* 0.0016
- *First-token latency:* 0.802
- *Cleanup time:* 0.171
- *Prompt tokens:* 397
- *Generation tokens:* 241
- *Total tokens:* 638
- *Prompt throughput (raw):* 495 tok/s
- *Generation throughput (raw):* 4.7 tok/s
- *Peak memory:* 39
- *Active memory:* 33
- *Cache memory:* 4.9
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.0057
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 227
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
◁think▷So, let's analyze the image. The image shows two tabby cats lying on a pink fabric surface. There are two remote controls, one near each cat. The cats have a mix of brown, black, and white fur. The lighting is high-contrast, making the colors very vivid. The scene is likely a cozy indoor setting, maybe a couch or bed. The cats appear relaxed, possibly sleeping or resting. The remotes are light blue with buttons, typical of TV or other electronic remotes. The overall style is a bit stylized with high saturation, giving it a vibrant look.◁/think▷The image depicts two tabby cats resting on a pink fabric surface. One cat is positioned on the left, with a light blue remote control near its head, while the other cat is on the right, with a similar remote control closer to its body. Both cats have a mix of brown, black, and white fur patterns. The scene has a high-contrast, stylized appearance with vivid colors, suggesting a cozy indoor setting, likely a couch or bed. The lighting emphasizes the cats' fur details and the texture of the fabric.&lt;|im_assistant|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷So, let's analyze the image. The image shows two tabby cats lying on a pink fabric surface. There are two remote controls, one near each cat. The cats have a mix of brown, black, and white fur. The lighting is high-contrast, making the colors very vivid. The scene is likely a cozy indoor setting, maybe a couch or bed. The cats appear relaxed, possibly sleeping or resting. The remotes are light blue with buttons, typical of TV or other electronic remotes. The overall style is a bit stylized with high saturation, giving it a vibrant look.◁/think▷The image depicts two tabby cats resting on a pink fabric surface. One cat is positioned on the left, with a light blue remote control near its head, while the other cat is on the right, with a similar remote control closer to its body. Both cats have a mix of brown, black, and white fur patterns. The scene has a high-contrast, stylized appearance with vivid colors, suggesting a cozy indoor setting, likely a couch or bed. The lighting emphasizes the cats' fur details and the texture of the fabric.<|im_assistant|>
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
- *Observations:* Unrecognised model control tokens remain visible
- *Model load time:* 3.44s
- *Generation time:* 2.11s
- *Total time:* 5.57s
- *Input validation time:* 0.00169
- *Prompt preparation time:* 0.02
- *First-token latency:* 0.291
- *Cleanup time:* 0.129
- *Prompt tokens:* 280
- *Generation tokens:* 88
- *Total tokens:* 368
- *Prompt throughput (raw):* 962 tok/s
- *Generation throughput (raw):* 53.4 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0156
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 66
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
&lt;channel|&gt;A high-angle eye-level shot shows two tabby cats sleeping on a pink blanket on a red couch. The cat on the left is smaller and lying on its side with its head turned up and its paws curled. The cat on the right is larger and is lying on its stomach with its head turned down. There are two remote controls on the couch next to the cats. The lighting is bright and even.
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>A high-angle eye-level shot shows two tabby cats sleeping on a pink blanket on a red couch. The cat on the left is smaller and lying on its side with its head turned up and its paws curled. The cat on the right is larger and is lying on its stomach with its head turned down. There are two remote controls on the couch next to the cats. The lighting is bright and even.
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
- *Model load time:* 3.17s
- *Generation time:* 1.78s
- *Total time:* 4.95s
- *Input validation time:* 0.00166
- *Prompt preparation time:* 0.00029
- *First-token latency:* 0.289
- *Cleanup time:* 0.124
- *Prompt tokens:* 280
- *Generation tokens:* 89
- *Total tokens:* 369
- *Prompt throughput (raw):* 970 tok/s
- *Generation throughput (raw):* 67.9 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0156
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 66
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
&lt;channel|&gt;A high-angle, medium shot shows two tabby cats sleeping on a pink blanket on a red couch. The cat on the left is smaller and is lying on its side with its head turned and its paws tucked in. The cat on the right is larger and is lying on its stomach with its head down and legs stretched out. Two remote controls are on the couch between the cats. The lighting is bright and even.
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>A high-angle, medium shot shows two tabby cats sleeping on a pink blanket on a red couch. The cat on the left is smaller and is lying on its side with its head turned and its paws tucked in. The cat on the right is larger and is lying on its stomach with its head down and legs stretched out. Two remote controls are on the couch between the cats. The lighting is bright and even.
```

</details>

</details>

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### mlx-community/paligemma2-3b-pt-896-4bit

<details>
<summary>Complete evidence: mlx-community/paligemma2-3b-pt-896-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response is unusually short
- *Model load time:* 1.13s
- *Generation time:* 1.12s
- *Total time:* 2.27s
- *Input validation time:* 0.00176
- *Prompt preparation time:* 0.000104
- *First-token latency:* 1.08
- *Cleanup time:* 0.0956
- *Prompt tokens:* 4,101
- *Generation tokens:* 2
- *Total tokens:* 4,103
- *Prompt throughput (raw):* 3,802 tok/s
- *Generation throughput (raw):* 87.8 tok/s
- *Peak memory:* 4.6
- *Active memory:* 1.7
- *Cache memory:* 1.7
- *Model-load active memory:* 1.74
- *Post-cleanup active memory:* 0.0241
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 26
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
Cat
</pre>

<details>
<summary>Exact raw output</summary>

```text
Cat
```

</details>

</details>

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

<details>
<summary>Complete evidence: mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response appears cut off at the token limit
- *Model load time:* 1.61s
- *Generation time:* 12.21s
- *Total time:* 13.83s
- *Input validation time:* 0.00161
- *Prompt preparation time:* 0.00658
- *First-token latency:* 0.91
- *Cleanup time:* 0.0842
- *Prompt tokens:* 1,329
- *Generation tokens:* 500
- *Total tokens:* 1,829
- *Prompt throughput (raw):* 1,460 tok/s
- *Generation throughput (raw):* 44.3 tok/s
- *Peak memory:* 14
- *Active memory:* 13
- *Cache memory:* 0.41
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.00207
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 414
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
The user asks: "Describe this image". The image is of two cats sleeping on a pink couch, with a remote control and a DVD case (or something) on the couch. The cats are both tabby, with similar patterns. They are lying on their sides, curled up, with paws extended. The remote is on the left side, the DVD case on the right side. The couch is pink. The cats appear to be sleeping. The cat on the left is lying on its side with its head turned slightly upward, its front paws stretched out, and its tail curled around its body. The cat on the right is lying on its side, with its head turned to the left, its front paws tucked under its chest, and its tail draped over its back. The remote is a typical TV remote with colored buttons. The DVD case is partially visible, showing a blue spine with some text. The couch is plush. The lighting is soft. The cats have a mix of brown, black, and tan stripes. The overall scene is cozy and relaxed.

We need to produce a description. The user didn't ask for anything else. So we can give a detailed description. The user is presumably just wanting a description. There's no disallowed content. So we can comply.

We should be mindful of the policy: no disallowed content. It's a normal image. So we can comply.

We can also mention the cat's positions relative to each other and the objects. The cat on the left is near the remote, the cat on the right near the DVD case. The couch is pink. The cats are tabby. The remote is white with black and colored buttons. The DVD case is blue with white text. The cats are sleeping. The scene is cozy.

We can also mention the cat's fur pattern: stripes, spots, etc. The cat's eyes are closed. The cat's ears are perked. The cat's paws are visible. The cat's tail is visible. The cat's whiskers are visible. The cat's nose is visible. The cat's ears are pointed. The cat's fur is fluffy.

We can also mention the couch's texture: plush, soft. The lighting is warm. The scene is likely in a living room.

We can also mention the remote is a typical TV remote with a power button, volume, channel, etc. The
</pre>

<details>
<summary>Exact raw output</summary>

```text
Here are my reasoning steps:
The user asks: "Describe this image". The image is of two cats sleeping on a pink couch, with a remote control and a DVD case (or something) on the couch. The cats are both tabby, with similar patterns. They are lying on their sides, curled up, with paws extended. The remote is on the left side, the DVD case on the right side. The couch is pink. The cats appear to be sleeping. The cat on the left is lying on its side with its head turned slightly upward, its front paws stretched out, and its tail curled around its body. The cat on the right is lying on its side, with its head turned to the left, its front paws tucked under its chest, and its tail draped over its back. The remote is a typical TV remote with colored buttons. The DVD case is partially visible, showing a blue spine with some text. The couch is plush. The lighting is soft. The cats have a mix of brown, black, and tan stripes. The overall scene is cozy and relaxed.

We need to produce a description. The user didn't ask for anything else. So we can give a detailed description. The user is presumably just wanting a description. There's no disallowed content. So we can comply.

We should be mindful of the policy: no disallowed content. It's a normal image. So we can comply.

We can also mention the cat's positions relative to each other and the objects. The cat on the left is near the remote, the cat on the right near the DVD case. The couch is pink. The cats are tabby. The remote is white with black and colored buttons. The DVD case is blue with white text. The cats are sleeping. The scene is cozy.

We can also mention the cat's fur pattern: stripes, spots, etc. The cat's eyes are closed. The cat's ears are perked. The cat's paws are visible. The cat's tail is visible. The cat's whiskers are visible. The cat's nose is visible. The cat's ears are pointed. The cat's fur is fluffy.

We can also mention the couch's texture: plush, soft. The lighting is warm. The scene is likely in a living room.

We can also mention the remote is a typical TV remote with a power button, volume, channel, etc. The
```

</details>

</details>

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response appears cut off at the token limit
- *Model load time:* 5.48s
- *Generation time:* 10.23s
- *Total time:* 15.72s
- *Input validation time:* 0.00166
- *Prompt preparation time:* 0.00774
- *First-token latency:* 2.51
- *Cleanup time:* 0.207
- *Prompt tokens:* 437
- *Generation tokens:* 500
- *Total tokens:* 937
- *Prompt throughput (raw):* 174 tok/s
- *Generation throughput (raw):* 64.9 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.06
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00256
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 230
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
Alright, let's get to work. The user wants a detailed description of this image, and I'm the one to provide it. First, I need to break down the request. The user has given me a complex prompt, a jumble of words that I need to decipher. It's a classic case of "gibberish," but I can handle it.

My initial step is to analyze the user's input. I see a jumble of words, a mix of cat breeds, colors, and objects. It's a mess, but I can see the core elements: cats, a couch, and a remote. The user wants a description, so I need to extract the meaning from this chaos.

Next, I'll sift through the text, pulling out the key elements. I'll identify the cats, the couch, and the remote. I'll also note the colors and positions. I'll ignore the gibberish, the nonsensical phrases.

Now, I'll synthesize the information. I'll start with the main subject: two cats on a couch. I'll describe their appearance, focusing on the colors and breeds. Then, I'll describe the setting: the couch, the remote, and the overall atmosphere. I'll make sure to mention the relaxed, cozy vibe.

Finally, I'll structure the description. I'll start with a general overview, then go into detail about the cats, the couch, and the remote. I'll use descriptive language to paint a picture. I'll make sure the description is clear, concise, and engaging. I'll also make sure to mention the "gibberish" in the prompt, but I won't let it dominate the description.

I'll review the final description to ensure it's accurate, comprehensive, and captures the essence of the image. I'll make sure it's well-written and easy to understand. I'll also make sure to mention the "gibberish" in the prompt, but I won't let it dominate the description.
&lt;/think&gt;

Based on the image provided, here is a description of the scene:

This is a heartwarming and cozy image of two cats relaxing together on a plush, red couch.

*   **
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get to work. The user wants a detailed description of this image, and I'm the one to provide it. First, I need to break down the request. The user has given me a complex prompt, a jumble of words that I need to decipher. It's a classic case of "gibberish," but I can handle it.

My initial step is to analyze the user's input. I see a jumble of words, a mix of cat breeds, colors, and objects. It's a mess, but I can see the core elements: cats, a couch, and a remote. The user wants a description, so I need to extract the meaning from this chaos.

Next, I'll sift through the text, pulling out the key elements. I'll identify the cats, the couch, and the remote. I'll also note the colors and positions. I'll ignore the gibberish, the nonsensical phrases.

Now, I'll synthesize the information. I'll start with the main subject: two cats on a couch. I'll describe their appearance, focusing on the colors and breeds. Then, I'll describe the setting: the couch, the remote, and the overall atmosphere. I'll make sure to mention the relaxed, cozy vibe.

Finally, I'll structure the description. I'll start with a general overview, then go into detail about the cats, the couch, and the remote. I'll use descriptive language to paint a picture. I'll make sure the description is clear, concise, and engaging. I'll also make sure to mention the "gibberish" in the prompt, but I won't let it dominate the description.

I'll review the final description to ensure it's accurate, comprehensive, and captures the essence of the image. I'll make sure it's well-written and easy to understand. I'll also make sure to mention the "gibberish" in the prompt, but I won't let it dominate the description.
</think>

Based on the image provided, here is a description of the scene:

This is a heartwarming and cozy image of two cats relaxing together on a plush, red couch.

*   **
```

</details>

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### mlx-community/Kimi-VL-A3B-Thinking-8bit

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response appears cut off at the token limit; Internal
  reasoning block appears incomplete
- *Model load time:* 2.39s
- *Generation time:* 7.74s
- *Total time:* 10.14s
- *Input validation time:* 0.00166
- *Prompt preparation time:* 0.00157
- *First-token latency:* 0.282
- *Cleanup time:* 0.192
- *Prompt tokens:* 397
- *Generation tokens:* 500
- *Total tokens:* 897
- *Prompt throughput (raw):* 1,408 tok/s
- *Generation throughput (raw):* 68.2 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.76
- *Model-load active memory:* 17.8
- *Post-cleanup active memory:* 0.00636
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 227
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
◁think▷Okay, let me try to figure out how to describe this image. First, I need to look at the image carefully. There are two cats on a pink surface, maybe a blanket or a bed. Both cats seem to be lying down. The one on the left is a tabby with gray and black stripes, and the one on the right is a darker tabby, maybe a bit bigger. They both look like they're sleeping or resting. There's a remote control near the left cat and another one near the right cat. The background is pink, which might be a couch or a bed. The image is a bit blurry, especially around the edges, which makes it look like a low-resolution or heavily compressed image. The colors are a bit oversaturated, especially the pink background. The cats' fur has some highlights, maybe from light reflecting off them. The cats seem to be in a relaxed position, maybe after playing or moving around since one is stretched out. The remotes are standard TV remotes, which adds a bit of context that this might be a cozy indoor setting. The image might have been taken in a living room. The lighting seems artificial, maybe from a lamp or overhead light. The cats' eyes aren't clearly visible because they're closed. The overall mood is calm and relaxed. I should mention the colors, the subjects (cats and remotes), the setting, and the image quality issues like blurriness and color saturation. Also, the composition with the remotes adds a bit of a humorous touch, like the cats are ignoring the remotes. Maybe mention the possible breeds of the cats, but I'm not sure. The image might have been edited or filtered, given the color enhancements. The pink background contrasts with the cats' fur colors. The cats' positions suggest they might have been moving before settling down. The remotes are standard, not specialized, so maybe they're in a typical living room. The image could be a candid shot capturing a quiet moment. The blurriness might be due to motion or a low-quality camera. The lighting is even, not harsh shadows, so maybe it's evening or night time. The cats seem comfortable, indicating they feel safe. The image might be meant to be cute or humorous because of the remotes next to them. The pink background makes the cats stand out. The image could be shared on social media or as a pet owner's photo. The cats' poses are natural, not posed.
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let me try to figure out how to describe this image. First, I need to look at the image carefully. There are two cats on a pink surface, maybe a blanket or a bed. Both cats seem to be lying down. The one on the left is a tabby with gray and black stripes, and the one on the right is a darker tabby, maybe a bit bigger. They both look like they're sleeping or resting. There's a remote control near the left cat and another one near the right cat. The background is pink, which might be a couch or a bed. The image is a bit blurry, especially around the edges, which makes it look like a low-resolution or heavily compressed image. The colors are a bit oversaturated, especially the pink background. The cats' fur has some highlights, maybe from light reflecting off them. The cats seem to be in a relaxed position, maybe after playing or moving around since one is stretched out. The remotes are standard TV remotes, which adds a bit of context that this might be a cozy indoor setting. The image might have been taken in a living room. The lighting seems artificial, maybe from a lamp or overhead light. The cats' eyes aren't clearly visible because they're closed. The overall mood is calm and relaxed. I should mention the colors, the subjects (cats and remotes), the setting, and the image quality issues like blurriness and color saturation. Also, the composition with the remotes adds a bit of a humorous touch, like the cats are ignoring the remotes. Maybe mention the possible breeds of the cats, but I'm not sure. The image might have been edited or filtered, given the color enhancements. The pink background contrasts with the cats' fur colors. The cats' positions suggest they might have been moving before settling down. The remotes are standard, not specialized, so maybe they're in a typical living room. The image could be a candid shot capturing a quiet moment. The blurriness might be due to motion or a low-quality camera. The lighting is even, not harsh shadows, so maybe it's evening or night time. The cats seem comfortable, indicating they feel safe. The image might be meant to be cute or humorous because of the remotes next to them. The pink background makes the cats stand out. The image could be shared on social media or as a pet owner's photo. The cats' poses are natural, not posed.
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
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit
- *Model load time:* 1.62s
- *Generation time:* 4.22s
- *Total time:* 5.86s
- *Input validation time:* 0.00165
- *Prompt preparation time:* 0.000169
- *First-token latency:* 0.229
- *Cleanup time:* 0.0974
- *Prompt tokens:* 264
- *Generation tokens:* 500
- *Total tokens:* 764
- *Prompt throughput (raw):* 1,151 tok/s
- *Generation throughput (raw):* 126 tok/s
- *Peak memory:* 5.8
- *Active memory:* 4.5
- *Cache memory:* 0.03
- *Model-load active memory:* 4.48
- *Post-cleanup active memory:* 0.0172
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 38
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
 this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this
</pre>

<details>
<summary>Exact raw output</summary>

```text
 this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this
```

</details>

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
- *Model load time:* 0.21s
- *Generation time:* -
- *Total time:* 0.22s
- *Input validation time:* 0.00174
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0622
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.00442
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

*Complete traceback:*

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11838, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11174, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 711, in _typed_mlx_vlm_load
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12091, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11853, in _run_model_generation
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

*Captured upstream output:*

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 54 files:   0%|          | 0/54 [00:00<?, ?it/s]
Fetching 54 files: 100%|##########| 54/54 [00:00<00:00, 5009.23it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[00:32:22] DEBUG    HF Cache Info for mlx-community/Inkling-Small-mlx-4bit: size=146358.0 MB, files=58
```

</details>

---
