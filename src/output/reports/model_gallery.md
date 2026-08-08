# Model Output Gallery

Generated on: 2026-08-08 22:59:20 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS    | Prefill/first s | Peak GB | Gen tok | Observations                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|------------|-----------------|---------|---------|-----------------------------------------------------------------------------------------------------|
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 7.20s   | 30.4 tok/s | 1.15            | 17      | 103     | none                                                                                                |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | 11.39s  | 13.7 tok/s | 2.23            | 15      | 89      | none                                                                                                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 7.61s   | 53.9 tok/s | 2.86            | 12      | 121     | none                                                                                                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              | 10.38s  | 43.8 tok/s | 3.20            | 12      | 171     | none                                                                                                |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 4.58s   | 156 tok/s  | 1.05            | 6.4     | 117     | none                                                                                                |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | 4.37s   | 52.6 tok/s | 0.31            | 9.4     | 100     | none                                                                                                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 83.67s  | 59.0 tok/s | 78.57           | 24      | 79      | none                                                                                                |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 87.08s  | 67.6 tok/s | 82.32           | 10      | 106     | none                                                                                                |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | 58.16s  | 19.2 tok/s | 39.19           | 70      | 118     | none                                                                                                |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 5.16s   | 104 tok/s  | 0.47            | 16      | 96      | none                                                                                                |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 12.50s  | 12.2 tok/s | 2.24            | 20      | 83      | none                                                                                                |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | 10.04s  | 18.8 tok/s | 2.85            | 15      | 82      | none                                                                                                |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable_with_caveats` | 1.29s   | 520 tok/s  | 0.04            | 1.2     | 107     | title/keyword constraints failed                                                                    |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | 16.82s  | 93.0 tok/s | 14.09           | 8.4     | 90      | title/keyword constraints failed                                                                    |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | 10.20s  | 26.8 tok/s | 2.85            | 22      | 110     | title/keyword constraints failed                                                                    |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | 11.63s  | 71.1 tok/s | 7.96            | 8.5     | 104     | title/keyword constraints failed                                                                    |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | 9.22s   | 30.3 tok/s | 1.24            | 18      | 130     | role tokens visible                                                                                 |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | 6.66s   | 43.2 tok/s | 0.93            | 11      | 105     | title/keyword constraints failed                                                                    |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable_with_caveats` | 114.54s | 35.0 tok/s | 94.10           | 74      | 107     | title/keyword constraints failed                                                                    |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable_with_caveats` | 107.13s | 198 tok/s  | 104.83          | 5.1     | 102     | title/keyword constraints failed; draft hints copied unchanged                                      |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | 28.00s  | 80.0 tok/s | 24.59           | 8.4     | 90      | title/keyword constraints failed                                                                    |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable_with_caveats` | 137.36s | 7.31 tok/s | 120.62          | 35      | 87      | title/keyword constraints failed                                                                    |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | 4.36s   | 122 tok/s  | 0.40            | 5.5     | 125     | title/keyword constraints failed                                                                    |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | 19.51s  | 11.5 tok/s | 1.81            | 18      | 155     | title/keyword constraints failed                                                                    |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `usable_with_caveats` | 7.02s   | 43.9 tok/s | 0.40            | 17      | 130     | title/keyword constraints failed                                                                    |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 17.05s  | 38.7 tok/s | 1.77            | 14      | 500     | missing required fields; echoes instructions; cut off at token limit                                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 29.93s  | 49.0 tok/s | 7.02            | 60      | 500     | repeated text; missing required fields; cut off at token limit                                      |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 2.73s   | 323 tok/s  | 0.10            | 2.1     | 130     | missing required fields; extra text before Title                                                    |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 22.88s  | 39.4 tok/s | 7.52            | 13      | 500     | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `unusable`            | 39.93s  | 26.2 tok/s | 24.12           | 78      | 92      | control tokens visible; missing required fields; extra text before Title                            |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | 127.49s | 4.45 tok/s | 2.92            | 40      | 500     | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `unusable`            | 2.63s   | 171 tok/s  | 0.12            | 3.8     | 93      | missing required fields                                                                             |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | 5.33s   | 182 tok/s  | 0.26            | 3.8     | 500     | missing required fields; cut off at token limit; incomplete thinking block                          |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | 35.65s  | 5.76 tok/s | 4.31            | 24      | 156     | missing required fields                                                                             |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | 33.56s  | 81.8 tok/s | 24.42           | 8.4     | 500     | missing required fields; echoes instructions; cut off at token limit                                |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | 69.07s  | 25.3 tok/s | 45.69           | 14      | 500     | repeated text; cut off at token limit; title/keyword constraints failed                             |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `unusable`            | 10.52s  | 17.7 tok/s | 0.37            | 29      | 81      | missing required fields; extra text before Title                                                    |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `unusable`            | 10.12s  | 19.4 tok/s | 0.38            | 28      | 82      | missing required fields; extra text before Title                                                    |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | 8.12s   | 45.3 tok/s | 5.06            | 9.7     | 52      | missing required fields                                                                             |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | 1.60s   | 317 tok/s  | 0.08            | 2.2     | 71      | missing required fields                                                                             |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | 22.96s  | 27.1 tok/s | 2.06            | 4.4     | 500     | repeated text; missing required fields; cut off at token limit                                      |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                                   | `not_evaluated`       | 0.59s   | -          | -               | -       | -       | none                                                                                                |
<!-- markdownlint-enable MD034 MD049 -->

## Resource Highlights

Fastest valid generation: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 520 tok/s

Average valid generation throughput: 76.5 tok/s

Lowest captured peak memory: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.2 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                        | Usability       | Observations                                                                                        |
|--------------------------------------------------------------------------------------------------------------|-----------------|-----------------------------------------------------------------------------------------------------|
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable`      | missing required fields; echoes instructions; cut off at token limit                                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable`      | repeated text; missing required fields; cut off at token limit                                      |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable`      | missing required fields; extra text before Title                                                    |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                   | `unusable`      | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                         | `unusable`      | control tokens visible; missing required fields; extra text before Title                            |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | `unusable`      | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                | `unusable`      | missing required fields                                                                             |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                 | `unusable`      | missing required fields; cut off at token limit; incomplete thinking block                          |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                | `unusable`      | missing required fields                                                                             |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                  | `unusable`      | missing required fields; echoes instructions; cut off at token limit                                |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                             |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | `unusable`      | missing required fields; extra text before Title                                                    |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | `unusable`      | missing required fields; extra text before Title                                                    |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                 | `unusable`      | missing required fields                                                                             |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable`      | missing required fields                                                                             |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable`      | repeated text; missing required fields; cut off at token limit                                      |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                        | `not_evaluated` | none                                                                                                |
<!-- markdownlint-enable MD034 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                           |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | Title: Speeding Boats on Ocean Horizon<br><br>Description: Two inflatable boats with passengers speed across the ocean, leaving white wakes under a clear blue sky, captured at 2026-08-07 17:17:16 UTC+01:00.<br><br>Keywords: Inflatable boat, Motorboat, People, Riding, Ocean, Horizon, Blue...                      |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | Title: Speeding Inflatable Boats on the Open Ocean<br><br>Description: Two inflatable boats with outboard motors speed across the ocean on a sunny day, leaving behind white wakes. The clear blue sky and distinct horizon line provide a picturesque backdrop.<br><br>Keywords: Ocean, Inflatab...                     |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>**Speeding Inflatable Boats Near Seaford, UK**<br><br>Description:<br>Two inflatable boats with outboard motors race across calm waters near Seaford, England, on a clear August evening in 2026, leaving pronounced white wakes under a vivid blue sky and distinct horizon.<br><br>Keywords...               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              | **Title:**<br>*Inflatable Boats Racing Near Seaford, UK (2026-08-07)*<br><br>**Description:**<br>On **2026-08-07 at 17:17 UTC+01:00**, two inflatable motorboats speed across calm waters near **Seaford, England (UK)**, leaving pronounced white wakes. Against a clear blue sky and distinct h...                     |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>**Two Speeding Motorboats on Clear Seas Near Seaford**<br><br>Description:<br>Two inflatable motorboats filled with passengers are racing across calm ocean waters under a bright blue sky, leaving pronounced white wakes. The scene captures a leisurely yet dynamic holiday moment i...                     |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | Title: "Inflatable Boats Speed Across the Ocean"<br><br>Description: On August 7, 2026, two inflatable boats with outboard motors are seen speeding across the ocean in Seaford, England, leaving white wakes behind them under a clear blue sky.<br><br>Keywords: inflatable boats, outboard mot...                     |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: Speeding Inflatable Boats on Blue Sea<br><br>Description: Two motorized inflatable boats carrying passengers speed across calm blue waters under a clear sky, leaving white wakes behind them.<br><br>Keywords: inflatable boat, motorboat, sea, ocean, water, people, passengers, speed,...                      |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | Title:<br>Two inflatable motorboats speeding across the sea at Seaford, UK.<br><br>Description:<br>Two inflatable motorboats with outboard motors speed across calm blue waters under a clear sky, leaving white wakes behind them, near the horizon line in Seaford, England, UK.<br><br>Keywords:<br>Inf...            |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | Title:<br>Two Inflatable Boats Speeding Across the Ocean Near Seaford<br><br>Description:<br>On 7 August 2026 at 17:17 UTC+1, two inflatable motorboats with outboard engines speed across the ocean off Seaford, England, leaving white wakes behind them under a clear blue sky and a distinct...                      |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: Two motorboats speeding across the blue sea<br>Description: Two inflatable motorboats speed across the blue ocean, leaving white wakes behind them under a clear sky. The scene captures a sense of action against a sharp horizon line near Seaford, England.<br>Keywords: blue sky...                           |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: Inflatable boats speeding off the coast of Seaford<br>Description: Two inflatable motorboats carrying passengers speed across the blue waters of the English Channel near Seaford, England, under a clear sky.<br>Keywords: Seaford, England, UK, inflatable boat, motorboat, ocean,...                           |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | Title:<br>Two Inflatable Boats Speeding Across the Sea<br><br>Description:<br>Two inflatable boats with outboard motors speed across the sea, leaving white wakes behind them under a clear blue sky with a distinct horizon line.<br><br>Keywords:<br>Inflatable boat, Motorboat, Sea, Blue sky, Horizon,...            |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable_with_caveats` | Title:<br>Seafront, Seaford, England, UK, GBR, Europe<br><br>Description:<br>Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.<br><br>Keywords:<br>Seafront, Seaford, England, UK, GBR, Euro...            |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | Title: Two boats on the sea<br><br>Description: Two inflatable motorboats speed across the blue sea, leaving white wakes, under a clear sky with a distinct horizon, with people on board enjoying the water.<br><br>Keywords: motorboat, inflatable boat, sea, people, sky, horizon, water, acti...                     |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | Title:<br>Two Inflatable Boats Racing Across the Ocean<br><br>Description:<br>Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, under a clear blue sky with a distinct horizon line.<br><br>Keywords:<br>Adobe Stock, inflatable boats, outboard motors,...             |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | Title: Two Inflatable Boats<br>Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.<br>Keywords: Two Inflatable Boats, outboard motors, speeding across ocean, whit...                           |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | Title: Inflatable Boats Speed Across the Ocean on a Clear Day.<br><br>Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. The boats are carrying several people, a...                           |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | Title: Speedboats Raging on Seaford's Shores<br><br>Description: Two inflatable motorboats race across the ocean, leaving white wakes behind them against a clear blue sky and distinct horizon, with passengers enjoying the thrilling ride on a sunny day in Seaford, England.<br><br>Keywords:...                     |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable_with_caveats` | Title: Two inflatable motorboats speeding across calm blue sea<br><br>Description: Two rigid inflatable boats carrying passengers cruise across the open sea under a clear sky, leaving white wakes behind them. The scene captures a bright, sunny day on the water with a distinct horizo...                           |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable_with_caveats` | Title:<br>Seafront, Seaford, England, UK, GBR, Europe<br><br>Description:<br>Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.<br><br>Keywords:<br>Adobe Stock, Any Vision, Blue sky, Driver...            |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | Title: Two boats on the sea<br><br>Description: Two inflatable motorboats speed across the blue sea, leaving white wakes, under a clear sky with a distinct horizon, with people on board enjoying the water.<br><br>Keywords: motorboat, inflatable boat, sea, people, sky, horizon, water, acti...                     |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable_with_caveats` | Title: Two Speedboats Racing Across the Sea<br><br>Description: Two motorised inflatable boats carrying passengers speed across the blue ocean, leaving white wakes behind them under a clear sky.<br><br>Keywords: Sea, Ocean, Boats, Motorboats, Inflatable boats, People, Passengers, Speeding...                     |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | Title: "Seafront Adventure"<br>Description: "Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. The boats are being ridden by people enjoying a holiday in England, Europe, with a...                              |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | Title: Seaford Ribs Speeding Across the Channel, August 2026<br><br>Description: Two black rigid inflatable boats (RIBs) with passengers are travelling at speed on the sea off Seaford, England, captured on 7th August 2026 at 17:17 UTC+01:00 under clear, bright conditions. Distinct w...                           |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `usable_with_caveats` | **Title:** Two Inflatable Boats Speed Across the Ocean<br><br>**Description:** Two blue inflatable boats, each with an outboard motor, are speeding across a vast, choppy blue ocean. White wakes trail behind them, contrasting against the deep blue water. The sky is a clear, pale blue...                           |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time: 2026-08-07 17:17:16 UTC+01:00. The descriptive hints: Title hint: Seafront, Seaford, England, UK, GBR, Europe. Descri...                              |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | Alright, let's get to work. I've got the image and the hints, and the task is to create some metadata. First, the title. "Two inflatable boats speeding across the sea" is a good start, but I need something more evocative. "Boats on the Seafront at Seaford" is a bit bland. "Spe...                                 |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | A concrete 5-10-word title: "Two inflatable boats speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line."<br>A 1-2-sentence factual description: "Two inflatable boats with outboard motors are speeding across the ocean, le...                              |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. Looking at the image: two inflatable boats, sea, action. So maybe "Inflatable Boats Speeding Over Ocean".<br><br>Next, the description: need to combine context (date/time, location...                     |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `unusable`            | <\|begin_of_box\|>Title: Two inflatable boats speeding across the ocean<br>Description: Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line, with people aboard each vessel.<br>Keywords:...                         |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | ◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.<br><br>First, the context says the capture date/time is 2026-08-07 17:17:16 UTC+01:00, but we...                           |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `unusable`            | Seafoam Adventure, Two Inflatable Boats, Clear Sky, Horizon, People, Motorboat, Adventure, Seaford, England, Europe, Action, Water, Adventure, Inflatable Boat, Motorboat, People, Riding, Sailing, Seafoam, Blue Sky, UK, Vehicles, Water, Action, Beautiful, 2026-08-07 17:17:16 UT...                                 |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | &lt;think&gt;<br>We need to create the catalogue metadata. First, the title should be concrete and 5-10 words. Then the description should combine relevant context with main visible subjects, setting, action, lighting, and distinctive details. Then the keywords should be 10-18 unique,...                         |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | Inflatable Boats Speeding Across Open Sea Near Seaford<br><br>Two inflatable motorboats with outboard engines race across a vast, dark blue ocean, leaving white wakes behind them. The boats carry multiple passengers and feature distinctive blue and white designs, with one displaying...                           |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The image shows two inflatable boats (motorboats) on the ocean, so maybe "Two motorboats speeding across the sea" but need to be concise. Let's see: "Two inflatable motorboats racing on the ocean"...                                 |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | Title:<br>Inflatable Boats Speeding Across the Ocean<br><br>Description:<br>Two inflatable motorboats, one labeled "Rapid," cruise across the open sea, leaving white wakes as they move, under a clear blue sky with a distinct horizon line, capturing a vibrant holiday scene.<br><br>Keywords: Seaf...               |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `unusable`            | <\|channel>thought<br><channel\|>Title: Two Inflatable Boats Speeding Across Seaford Sea<br>Description: Two inflatable motorboats with outboard engines speed across the deep blue sea, leaving white wakes under a clear sky.<br>Keywords: Seaford, England, UK, ocean, inflatable boat, motor...                      |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `unusable`            | <\|channel>thought<br><channel\|>Title: Two Inflatable Boats Speeding Across the Seaford Sea<br>Description: Two inflatable boats with outboard motors speed across the blue ocean water, leaving white wakes behind them under a clear sky.<br>Keywords: Seaford, England, UK, ocean, inflatabl...                      |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | Seafront, Seaford, England, UK, GBR, Europe, Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.                                                                                                             |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | Title: "Seaford in Action"<br>Description: A serene scene of two inflatable boats, one driving away from the other, captured in motion on a clear blue sky. The boats are driving on a calm sea, with a distinct horizon line in the background. The image is a testament to the beauty...                               |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | - Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>-... |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                                   | `not_evaluated`       | Model loading failed: Received 362 parameters not in model:<br>audio_tower.encoder.biases,<br>audio_tower.encoder.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.biases,<br>language_model.model.layers.10.mlp.experts.down_proj.scales,<br>language_model.model.layers.10.mlp.expe...                  |
<!-- markdownlint-enable MD034 MD049 -->

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

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### mlx-community/InternVL3-8B-bf16

<details>
<summary>Complete evidence: mlx-community/InternVL3-8B-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type internvl_chat)
- *Model load time:* 1.64s
- *Generation time:* 5.20s
- *Total time:* 7.20s
- *Input validation time:* 0.361
- *Prompt preparation time:* 0.00195
- *First-token latency:* 1.15
- *Cleanup time:* 0.144
- *Prompt tokens:* 2,610
- *Generation tokens:* 103
- *Total tokens:* 2,713
- *Prompt throughput (raw):* 2,270 tok/s
- *Generation throughput (raw):* 30.4 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.21
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.00302
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Speeding Boats on Ocean Horizon

Description: Two inflatable boats with passengers speed across the ocean, leaving white wakes under a clear blue sky, captured at 2026-08-07 17:17:16 UTC+01:00.

Keywords: Inflatable boat, Motorboat, People, Riding, Ocean, Horizon, Blue sky, Clear weather, Action, Water, Wake, Holiday, Europe, UK, Seaford, Vehicles
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
- *Model load time:* 1.69s
- *Generation time:* 9.28s
- *Total time:* 11.39s
- *Input validation time:* 0.415
- *Prompt preparation time:* 0.00174
- *First-token latency:* 2.23
- *Cleanup time:* 0.155
- *Prompt tokens:* 297
- *Generation tokens:* 89
- *Total tokens:* 386
- *Prompt throughput (raw):* 133 tok/s
- *Generation throughput (raw):* 13.7 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.59
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.00433
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Model load time:* 1.51s
- *Generation time:* 5.68s
- *Total time:* 7.61s
- *Input validation time:* 0.406
- *Prompt preparation time:* 0.00924
- *First-token latency:* 2.86
- *Cleanup time:* 0.187
- *Prompt tokens:* 2,578
- *Generation tokens:* 121
- *Total tokens:* 2,699
- *Prompt throughput (raw):* 900 tok/s
- *Generation throughput (raw):* 53.9 tok/s
- *Peak memory:* 12
- *Active memory:* 8.0
- *Cache memory:* 0.51
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.00511
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-nvfp4</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.03s
- *Generation time:* 7.66s
- *Total time:* 10.38s
- *Input validation time:* 0.665
- *Prompt preparation time:* 0.000279
- *First-token latency:* 3.2
- *Cleanup time:* 0.234
- *Prompt tokens:* 2,578
- *Generation tokens:* 171
- *Total tokens:* 2,749
- *Prompt throughput (raw):* 807 tok/s
- *Generation throughput (raw):* 43.8 tok/s
- *Peak memory:* 12
- *Active memory:* 8.4
- *Cache memory:* 0.51
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00538
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
**Title:**
*Inflatable Boats Racing Near Seaford, UK (2026-08-07)*

**Description:**
On **2026-08-07 at 17:17 UTC+01:00**, two inflatable motorboats speed across calm waters near **Seaford, England (UK)**, leaving pronounced white wakes. Against a clear blue sky and distinct horizon, passengers appear engaged, showcasing dynamic movement and bright daylight conditions.

**Keywords:**
inflatable motorboats, Seaford, England, UK, Europe, 2026-08-07, racing, ocean, white wakes, clear blue sky, horizon, passengers, bright daylight, outboard motors, coastal waters, dynamic movement, horizon line, summer activity
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
- *Model load time:* 1.56s
- *Generation time:* 2.34s
- *Total time:* 4.58s
- *Input validation time:* 0.663
- *Prompt preparation time:* 0.0103
- *First-token latency:* 1.05
- *Cleanup time:* 0.125
- *Prompt tokens:* 2,577
- *Generation tokens:* 117
- *Total tokens:* 2,694
- *Prompt throughput (raw):* 2,449 tok/s
- *Generation throughput (raw):* 156 tok/s
- *Peak memory:* 6.4
- *Active memory:* 2.8
- *Cache memory:* 0.35
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00564
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
**Two Speeding Motorboats on Clear Seas Near Seaford**

Description:
Two inflatable motorboats filled with passengers are racing across calm ocean waters under a bright blue sky, leaving pronounced white wakes. The scene captures a leisurely yet dynamic holiday moment in a coastal setting, with clear lighting and a sharp horizon.

Keywords:
Adobe Stock, action, blue sky, England, Europe, holiday, inflatable boat, motorboat, ocean, people, riding, Seaford, UK, water, vehicles, clear skies, distinct wakes
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
- *Model load time:* 0.99s
- *Generation time:* 2.71s
- *Total time:* 4.37s
- *Input validation time:* 0.661
- *Prompt preparation time:* 0.00124
- *First-token latency:* 0.312
- *Cleanup time:* 0.104
- *Prompt tokens:* 1,112
- *Generation tokens:* 100
- *Total tokens:* 1,212
- *Prompt throughput (raw):* 3,560 tok/s
- *Generation throughput (raw):* 52.6 tok/s
- *Peak memory:* 9.4
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00714
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Generation time:* 80.72s
- *Total time:* 83.67s
- *Input validation time:* 0.443
- *Prompt preparation time:* 0.0142
- *First-token latency:* 78.6
- *Cleanup time:* 0.179
- *Prompt tokens:* 16,643
- *Generation tokens:* 79
- *Total tokens:* 16,722
- *Prompt throughput (raw):* 212 tok/s
- *Generation throughput (raw):* 59.0 tok/s
- *Peak memory:* 24
- *Active memory:* 21
- *Cache memory:* 0.88
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.00859
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Speeding Inflatable Boats on Blue Sea

Description: Two motorized inflatable boats carrying passengers speed across calm blue waters under a clear sky, leaving white wakes behind them.

Keywords: inflatable boat, motorboat, sea, ocean, water, people, passengers, speed, wake, blue sky, horizon, England, UK, Europe, action, vehicles, sailing, holiday
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
- *Model load time:* 1.66s
- *Generation time:* 84.89s
- *Total time:* 87.08s
- *Input validation time:* 0.517
- *Prompt preparation time:* 0.000371
- *First-token latency:* 82.3
- *Cleanup time:* 0.155
- *Prompt tokens:* 16,643
- *Generation tokens:* 106
- *Total tokens:* 16,749
- *Prompt throughput (raw):* 202 tok/s
- *Generation throughput (raw):* 67.6 tok/s
- *Peak memory:* 10
- *Active memory:* 6.6
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00911
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two inflatable motorboats speeding across the sea at Seaford, UK.

Description:
Two inflatable motorboats with outboard motors speed across calm blue waters under a clear sky, leaving white wakes behind them, near the horizon line in Seaford, England, UK.

Keywords:
Inflatable boat, motorboat, outboard motor, wake, blue sky, horizon, people, riding, action, Seaford, England, UK, Europe, holiday, water, vehicles, speed
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:  
Two inflatable motorboats speeding across the sea at Seaford, UK.

Description:  
Two inflatable motorboats with outboard motors speed across calm blue waters under a clear sky, leaving white wakes behind them, near the horizon line in Seaford, England, UK.

Keywords:  
Inflatable boat, motorboat, outboard motor, wake, blue sky, horizon, people, riding, action, Seaford, England, UK, Europe, holiday, water, vehicles, speed
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
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 11.51s
- *Generation time:* 46.14s
- *Total time:* 58.16s
- *Input validation time:* 0.477
- *Prompt preparation time:* 0.0141
- *First-token latency:* 39.2
- *Cleanup time:* 0.735
- *Prompt tokens:* 3,478
- *Generation tokens:* 118
- *Total tokens:* 3,596
- *Prompt throughput (raw):* 88.7 tok/s
- *Generation throughput (raw):* 19.2 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.41
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.01
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two Inflatable Boats Speeding Across the Ocean Near Seaford

Description:
On 7 August 2026 at 17:17 UTC+1, two inflatable motorboats with outboard engines speed across the ocean off Seaford, England, leaving white wakes behind them under a clear blue sky and a distinct horizon line.

Keywords:
Inflatable boat, Motorboat, Seaford, England, UK, Europe, Blue sky, Horizon, Ocean, Water, People, Action, Holiday, Sailing, Riding, Vehicles, Driver, Adobe Stock
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
- *Model load time:* 2.70s
- *Generation time:* 1.89s
- *Total time:* 5.16s
- *Input validation time:* 0.533
- *Prompt preparation time:* 0.0218
- *First-token latency:* 0.466
- *Cleanup time:* 0.118
- *Prompt tokens:* 588
- *Generation tokens:* 96
- *Total tokens:* 684
- *Prompt throughput (raw):* 1,263 tok/s
- *Generation throughput (raw):* 104 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.20
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0119
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Model load time:* 2.59s
- *Generation time:* 9.58s
- *Total time:* 12.50s
- *Input validation time:* 0.322
- *Prompt preparation time:* 0.000288
- *First-token latency:* 2.24
- *Cleanup time:* 0.137
- *Prompt tokens:* 588
- *Generation tokens:* 83
- *Total tokens:* 671
- *Prompt throughput (raw):* 262 tok/s
- *Generation throughput (raw):* 12.2 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.77
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0124
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Generation time:* 7.68s
- *Total time:* 10.04s
- *Input validation time:* 0.611
- *Prompt preparation time:* 0.00423
- *First-token latency:* 2.85
- *Cleanup time:* 0.166
- *Prompt tokens:* 2,649
- *Generation tokens:* 82
- *Total tokens:* 2,731
- *Prompt throughput (raw):* 930 tok/s
- *Generation throughput (raw):* 18.8 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.50
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0144
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

<details>
<summary>Complete evidence: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Duplicate keywords: seaford, uk
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.38s
- *Generation time:* 0.60s
- *Total time:* 1.29s
- *Input validation time:* 0.302
- *Prompt preparation time:* 0.00648
- *First-token latency:* 0.0389
- *Cleanup time:* 0.0615
- *Prompt tokens:* 385
- *Generation tokens:* 107
- *Total tokens:* 492
- *Prompt throughput (raw):* 9,903 tok/s
- *Generation throughput (raw):* 520 tok/s
- *Peak memory:* 1.2
- *Active memory:* 0.90
- *Cache memory:* 0.02
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000149
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Observations:* Keyword list has 20 terms (requested 10-18); Duplicate
  keywords: water
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.66s
- *Generation time:* 15.86s
- *Total time:* 16.82s
- *Input validation time:* 0.292
- *Prompt preparation time:* 0.00949
- *First-token latency:* 14.1
- *Cleanup time:* 0.0862
- *Prompt tokens:* 16,628
- *Generation tokens:* 90
- *Total tokens:* 16,718
- *Prompt throughput (raw):* 1,180 tok/s
- *Generation throughput (raw):* 93.0 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00046
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two boats on the sea

Description: Two inflatable motorboats speed across the blue sea, leaving white wakes, under a clear sky with a distinct horizon, with people on board enjoying the water.

Keywords: motorboat, inflatable boat, sea, people, sky, horizon, water, action, blue sky, England, UK, Europe, holiday, vehicles, seafront, Seaford, Sailing, Riding, Water, beautiful
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
- *Observations:* Keyword list has 24 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.19s
- *Generation time:* 7.56s
- *Total time:* 10.20s
- *Input validation time:* 0.431
- *Prompt preparation time:* 0.00869
- *First-token latency:* 2.85
- *Cleanup time:* 0.106
- *Prompt tokens:* 2,045
- *Generation tokens:* 110
- *Total tokens:* 2,155
- *Prompt throughput (raw):* 718 tok/s
- *Generation throughput (raw):* 26.8 tok/s
- *Peak memory:* 22
- *Active memory:* 18
- *Cache memory:* 0.42
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.000984
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Observations:* Title has 3 words (requested 5-10); Keyword list has 21
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.41s
- *Generation time:* 9.90s
- *Total time:* 11.63s
- *Input validation time:* 0.304
- *Prompt preparation time:* 0.00831
- *First-token latency:* 7.96
- *Cleanup time:* 0.145
- *Prompt tokens:* 6,330
- *Generation tokens:* 104
- *Total tokens:* 6,434
- *Prompt throughput (raw):* 796 tok/s
- *Generation throughput (raw):* 71.1 tok/s
- *Peak memory:* 8.5
- *Active memory:* 6.8
- *Cache memory:* 0.46
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00213
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Inflatable Boats
Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.
Keywords: Two Inflatable Boats, outboard motors, speeding across ocean, white wakes, clear blue sky, distinct horizon line, inflatable boat, motorboat, people, riding, Sailing, Seaford, England, UK, Europe, Holiday, Horizon, Sky, Water, action, beautiful
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
- *Model load time:* 2.26s
- *Generation time:* 6.24s
- *Total time:* 9.22s
- *Input validation time:* 0.706
- *Prompt preparation time:* 0.00209
- *First-token latency:* 1.24
- *Cleanup time:* 0.124
- *Prompt tokens:* 2,608
- *Generation tokens:* 130
- *Total tokens:* 2,738
- *Prompt throughput (raw):* 2,104 tok/s
- *Generation throughput (raw):* 30.3 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.0027
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Inflatable Boats Speed Across the Ocean on a Clear Day.

Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. The boats are carrying several people, and the water is calm with no visible waves. The sky is clear with no clouds, and the sun is not visible, suggesting it might be early morning or late afternoon.

Keywords: inflatable boats, ocean, outboard motors, white wakes, clear blue sky, horizon line, people, calm water, clear sky, no clouds, early morning, late afternoon.&lt;end_of_utterance&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Title: Inflatable Boats Speed Across the Ocean on a Clear Day.

Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. The boats are carrying several people, and the water is calm with no visible waves. The sky is clear with no clouds, and the sun is not visible, suggesting it might be early morning or late afternoon.

Keywords: inflatable boats, ocean, outboard motors, white wakes, clear blue sky, horizon line, people, calm water, clear sky, no clouds, early morning, late afternoon.<end_of_utterance>
```

</details>

</details>

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### mlx-community/Molmo-7B-D-0924-8bit

<details>
<summary>Complete evidence: mlx-community/Molmo-7B-D-0924-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type molmo)
- *Model load time:* 1.61s
- *Generation time:* 4.51s
- *Total time:* 6.66s
- *Input validation time:* 0.524
- *Prompt preparation time:* 0.0021
- *First-token latency:* 0.935
- *Cleanup time:* 0.293
- *Prompt tokens:* 1,498
- *Generation tokens:* 105
- *Total tokens:* 1,603
- *Prompt throughput (raw):* 1,603 tok/s
- *Generation throughput (raw):* 43.2 tok/s
- *Peak memory:* 11
- *Active memory:* 9.0
- *Cache memory:* 0.14
- *Model-load active memory:* 9.04
- *Post-cleanup active memory:* 0.00595
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### mlx-community/Ornith-1.0-35B-bf16

<details>
<summary>Complete evidence: mlx-community/Ornith-1.0-35B-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 14.59s
- *Generation time:* 99.24s
- *Total time:* 114.54s
- *Input validation time:* 0.677
- *Prompt preparation time:* 0.0162
- *First-token latency:* 94.1
- *Cleanup time:* 0.79
- *Prompt tokens:* 16,643
- *Generation tokens:* 107
- *Total tokens:* 16,750
- *Prompt throughput (raw):* 177 tok/s
- *Generation throughput (raw):* 35.0 tok/s
- *Peak memory:* 74
- *Active memory:* 70
- *Cache memory:* 0.88
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.00708
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Observations:* Keyword list has 20 terms (requested 10-18); Title,
  Description and Keywords copy all supplied hints unchanged
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.57s
- *Generation time:* 106.21s
- *Total time:* 107.13s
- *Input validation time:* 0.335
- *Prompt preparation time:* 0.00422
- *First-token latency:* 105
- *Cleanup time:* 0.112
- *Prompt tokens:* 16,639
- *Generation tokens:* 102
- *Total tokens:* 16,741
- *Prompt throughput (raw):* 159 tok/s
- *Generation throughput (raw):* 198 tok/s
- *Peak memory:* 5.1
- *Active memory:* 1.3
- *Cache memory:* 0.87
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.00746
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Observations:* Keyword list has 20 terms (requested 10-18); Duplicate
  keywords: water
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.84s
- *Generation time:* 26.72s
- *Total time:* 28.00s
- *Input validation time:* 0.427
- *Prompt preparation time:* 0.000291
- *First-token latency:* 24.6
- *Cleanup time:* 0.158
- *Prompt tokens:* 16,628
- *Generation tokens:* 90
- *Total tokens:* 16,718
- *Prompt throughput (raw):* 676 tok/s
- *Generation throughput (raw):* 80.0 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00777
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two boats on the sea

Description: Two inflatable motorboats speed across the blue sea, leaving white wakes, under a clear sky with a distinct horizon, with people on board enjoying the water.

Keywords: motorboat, inflatable boat, sea, people, sky, horizon, water, action, blue sky, England, UK, Europe, holiday, vehicles, seafront, Seaford, Sailing, Riding, Water, beautiful
</pre>

</details>

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### mlx-community/Qwen3.6-27B-mxfp8

<details>
<summary>Complete evidence: mlx-community/Qwen3.6-27B-mxfp8</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 3.34s
- *Generation time:* 133.46s
- *Total time:* 137.36s
- *Input validation time:* 0.506
- *Prompt preparation time:* 0.0146
- *First-token latency:* 121
- *Cleanup time:* 0.316
- *Prompt tokens:* 16,643
- *Generation tokens:* 87
- *Total tokens:* 16,730
- *Prompt throughput (raw):* 138 tok/s
- *Generation throughput (raw):* 7.31 tok/s
- *Peak memory:* 35
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.00962
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Speedboats Racing Across the Sea

Description: Two motorised inflatable boats carrying passengers speed across the blue ocean, leaving white wakes behind them under a clear sky.

Keywords: Sea, Ocean, Boats, Motorboats, Inflatable boats, People, Passengers, Speeding, Wakes, Blue water, Clear sky, Horizon, Summer, Daytime, England, UK, Europe, Action, Water sports
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
- *Model load time:* 0.87s
- *Generation time:* 2.73s
- *Total time:* 4.36s
- *Input validation time:* 0.75
- *Prompt preparation time:* 0.00462
- *First-token latency:* 0.4
- *Cleanup time:* 0.128
- *Prompt tokens:* 411
- *Generation tokens:* 125
- *Total tokens:* 536
- *Prompt throughput (raw):* 1,028 tok/s
- *Generation throughput (raw):* 122 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.19
- *Model-load active memory:* 4.5
- *Post-cleanup active memory:* 0.00973
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Observations:* Keyword list has 24 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 3.06s
- *Generation time:* 15.76s
- *Total time:* 19.51s
- *Input validation time:* 0.662
- *Prompt preparation time:* 0.00415
- *First-token latency:* 1.81
- *Cleanup time:* 0.319
- *Prompt tokens:* 579
- *Generation tokens:* 155
- *Total tokens:* 734
- *Prompt throughput (raw):* 320 tok/s
- *Generation throughput (raw):* 11.5 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.44
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0108
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Model load time:* 2.48s
- *Generation time:* 3.83s
- *Total time:* 7.02s
- *Input validation time:* 0.69
- *Prompt preparation time:* 0.00339
- *First-token latency:* 0.404
- *Cleanup time:* 0.223
- *Prompt tokens:* 578
- *Generation tokens:* 130
- *Total tokens:* 708
- *Prompt throughput (raw):* 1,431 tok/s
- *Generation throughput (raw):* 43.9 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0114
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Model load time:* 1.58s
- *Generation time:* 15.14s
- *Total time:* 17.05s
- *Input validation time:* 0.317
- *Prompt preparation time:* 0.00682
- *First-token latency:* 1.77
- *Cleanup time:* 0.148
- *Prompt tokens:* 2,740
- *Generation tokens:* 500
- *Total tokens:* 3,240
- *Prompt throughput (raw):* 1,549 tok/s
- *Generation throughput (raw):* 38.7 tok/s
- *Peak memory:* 14
- *Active memory:* 13
- *Cache memory:* 0.67
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.000722
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Here are my reasoning steps:
We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time: 2026-08-07 17:17:16 UTC+01:00. The descriptive hints: Title hint: Seafront, Seaford, England, UK, GBR, Europe. Description hint: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. Keyword hints: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful.

We need to produce:

- a concrete 5-10-word title (so 5-10 words). Should be British English. Should reflect the main subject. Possibly "Seaford Inflatable Boats Speeding Across the Sea". That's 7 words? Let's count: Seaford (1) Inflatable (2) Boats (3) Speeding (4) Across (5) the (6) Sea (7). That's 7 words. Good.

- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details. Should be 1-2 sentences. Should mention capture date/time? The instruction: "Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible." So we can incorporate the date/time in description, but not claim it's visible. So we can say "Captured on 7 August 2026 at 17:17 UTC+01, the image shows two inflatable motorboats racing across the open sea off Seaford, leaving white wakes behind them under a clear blue sky with a distinct horizon." That's one sentence. Or two sentences: "Captured on 7 August 2026 at 17:17 UTC+01, the image shows two inflatable motorboats racing across the open sea off Seaford. They leave white wakes behind them under a clear blue sky with a distinct horizon." That's two sentences. That's fine.

-
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
- *Model load time:* 11.50s
- *Generation time:* 18.07s
- *Total time:* 29.93s
- *Input validation time:* 0.335
- *Prompt preparation time:* 0.0119
- *First-token latency:* 7.02
- *Cleanup time:* 0.297
- *Prompt tokens:* 1,590
- *Generation tokens:* 500
- *Total tokens:* 2,090
- *Prompt throughput (raw):* 226 tok/s
- *Generation throughput (raw):* 49.0 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.14
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.0012
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Model load time:* 0.61s
- *Generation time:* 1.70s
- *Total time:* 2.73s
- *Input validation time:* 0.417
- *Prompt preparation time:* 0.00167
- *First-token latency:* 0.0984
- *Cleanup time:* 0.0985
- *Prompt tokens:* 319
- *Generation tokens:* 130
- *Total tokens:* 449
- *Prompt throughput (raw):* 3,241 tok/s
- *Generation throughput (raw):* 323 tok/s
- *Peak memory:* 2.1
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.00151
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Observations:* Missing or empty fields: Description, Keywords; Extra text
  appears before the Title field; Response appears cut off at the token limit;
  Internal reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.78s
- *Generation time:* 20.73s
- *Total time:* 22.88s
- *Input validation time:* 0.35
- *Prompt preparation time:* 0.00171
- *First-token latency:* 7.52
- *Cleanup time:* 0.144
- *Prompt tokens:* 6,326
- *Generation tokens:* 500
- *Total tokens:* 6,826
- *Prompt throughput (raw):* 841 tok/s
- *Generation throughput (raw):* 39.4 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.41
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.00182
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. Looking at the image: two inflatable boats, sea, action. So maybe "Inflatable Boats Speeding Over Ocean".

Next, the description: need to combine context (date/time, location) with main subject, setting, action, lighting, distinctive details. The image has two inflatable boats, ocean, clear blue sky, horizon, people. So something like "Two inflatable boats with outboard motors speed across the ocean on 2026-08-07, leaving white wakes under a clear blue sky with a distinct horizon." Wait, need to make sure it's 1-2 sentences. Let's check the hints: description hint says "Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line." So combine with context. So "On 2026-08-07, two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them under a clear blue sky with a distinct horizon."

Then keywords: need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (maybe not, since it's not visible), Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful. But need to pick unique ones. Let's list: Inflatable boat, Motorboat, Ocean, Seaford, England, UK, Europe, People, Blue sky, Horizon, Water, Action, Vehicles, Holiday, Sky, Driver, Wakes. Wait, need to make sure they are unique. Let's check the hints and image. The image has inflatable boats, motorboats, ocean, seaford, england, uk, europe, people, blue sky, horizon, water, action, vehicles, holiday, sky, driver. So keywords: Inflatable boat, Motorboat, Ocean, Seaford, England, UK, Europe, People, Blue sky, Horizon, Water, Action, Vehicles, Holiday, Sky, Driver, Wakes. That's around 16, which is within 10-18.

Now, let's structure each part:

Title: "Inflatable Boats Speeding
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. Looking at the image: two inflatable boats, sea, action. So maybe "Inflatable Boats Speeding Over Ocean". 

Next, the description: need to combine context (date/time, location) with main subject, setting, action, lighting, distinctive details. The image has two inflatable boats, ocean, clear blue sky, horizon, people. So something like "Two inflatable boats with outboard motors speed across the ocean on 2026-08-07, leaving white wakes under a clear blue sky with a distinct horizon." Wait, need to make sure it's 1-2 sentences. Let's check the hints: description hint says "Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line." So combine with context. So "On 2026-08-07, two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them under a clear blue sky with a distinct horizon." 

Then keywords: need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (maybe not, since it's not visible), Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful. But need to pick unique ones. Let's list: Inflatable boat, Motorboat, Ocean, Seaford, England, UK, Europe, People, Blue sky, Horizon, Water, Action, Vehicles, Holiday, Sky, Driver, Wakes. Wait, need to make sure they are unique. Let's check the hints and image. The image has inflatable boats, motorboats, ocean, seaford, england, uk, europe, people, blue sky, horizon, water, action, vehicles, holiday, sky, driver. So keywords: Inflatable boat, Motorboat, Ocean, Seaford, England, UK, Europe, People, Blue sky, Horizon, Water, Action, Vehicles, Holiday, Sky, Driver, Wakes. That's around 16, which is within 10-18. 

Now, let's structure each part:

Title: "Inflatable Boats Speeding
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
- *Model load time:* 11.24s
- *Generation time:* 28.17s
- *Total time:* 39.93s
- *Input validation time:* 0.499
- *Prompt preparation time:* 0.00708
- *First-token latency:* 24.1
- *Cleanup time:* 0.484
- *Prompt tokens:* 6,330
- *Generation tokens:* 92
- *Total tokens:* 6,422
- *Prompt throughput (raw):* 262 tok/s
- *Generation throughput (raw):* 26.2 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00244
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;|begin_of_box|&gt;Title: Two inflatable boats speeding across the ocean
Description: Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line, with people aboard each vessel.
Keywords: inflatable boat, motorboat, ocean, speeding, white wakes, clear blue sky, horizon line, people, boats, water, action, sea, seaford, england, uk, europe&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Two inflatable boats speeding across the ocean
Description: Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line, with people aboard each vessel.
Keywords: inflatable boat, motorboat, ocean, speeding, white wakes, clear blue sky, horizon line, people, boats, water, action, sea, seaford, england, uk, europe<|end_of_box|>
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
- *Model load time:* 7.49s
- *Generation time:* 119.50s
- *Total time:* 127.49s
- *Input validation time:* 0.475
- *Prompt preparation time:* 0.00186
- *First-token latency:* 2.92
- *Cleanup time:* 0.442
- *Prompt tokens:* 1,313
- *Generation tokens:* 500
- *Total tokens:* 1,813
- *Prompt throughput (raw):* 449 tok/s
- *Generation throughput (raw):* 4.45 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.6
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00367
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### mlx-community/LFM2.5-VL-1.6B-bf16

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.72s
- *Generation time:* 1.27s
- *Total time:* 2.63s
- *Input validation time:* 0.631
- *Prompt preparation time:* 0.00621
- *First-token latency:* 0.116
- *Cleanup time:* 0.128
- *Prompt tokens:* 571
- *Generation tokens:* 93
- *Total tokens:* 664
- *Prompt throughput (raw):* 4,906 tok/s
- *Generation throughput (raw):* 171 tok/s
- *Peak memory:* 3.8
- *Active memory:* 3.2
- *Cache memory:* 0.02
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.0038
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Model load time:* 1.10s
- *Generation time:* 3.72s
- *Total time:* 5.33s
- *Input validation time:* 0.482
- *Prompt preparation time:* 0.0114
- *First-token latency:* 0.26
- *Cleanup time:* 0.118
- *Prompt tokens:* 920
- *Generation tokens:* 500
- *Total tokens:* 1,420
- *Prompt throughput (raw):* 3,544 tok/s
- *Generation throughput (raw):* 182 tok/s
- *Peak memory:* 3.8
- *Active memory:* 2.3
- *Cache memory:* 0.06
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00485
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Model load time:* 2.58s
- *Generation time:* 32.32s
- *Total time:* 35.65s
- *Input validation time:* 0.732
- *Prompt preparation time:* 0.00579
- *First-token latency:* 4.31
- *Cleanup time:* 0.35
- *Prompt tokens:* 3,111
- *Generation tokens:* 156
- *Total tokens:* 3,267
- *Prompt throughput (raw):* 722 tok/s
- *Generation throughput (raw):* 5.76 tok/s
- *Peak memory:* 24
- *Active memory:* 19
- *Cache memory:* 5.0
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 0.00657
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
  Response repeats the task instructions instead of only returning the
  requested fields; Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 1.03s
- *Generation time:* 31.90s
- *Total time:* 33.56s
- *Input validation time:* 0.597
- *Prompt preparation time:* 0.0222
- *First-token latency:* 24.4
- *Cleanup time:* 0.171
- *Prompt tokens:* 16,630
- *Generation tokens:* 500
- *Total tokens:* 17,130
- *Prompt throughput (raw):* 681 tok/s
- *Generation throughput (raw):* 81.8 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.4
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00808
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The image shows two inflatable boats (motorboats) on the ocean, so maybe "Two motorboats speeding across the sea" but need to be concise. Let's see: "Two inflatable motorboats racing on the ocean" – that's 8 words, good.

Next, the description. It has to combine context with visible subject, setting, action, lighting, details. Context is capture date/time, but we don't claim that. The main subject is two inflatable boats with outboard motors, speeding, leaving wakes, clear blue sky, horizon line. So: "Two inflatable motorboats with outboard motors speed across the ocean, leaving white wakes, under a clear blue sky with distinct horizon line, on a calm sea."

Wait, the context says capture date/time is 2026-08-07 17:17:16 UTC+01:00, but we don't include that in the description. The descriptive hints mention "clear blue sky" which is visible, "distinct horizon line" – the image has a horizon line, so that's there. The boats are speeding, so "speeding" is key. Let's check the hints: "Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line." So the description should include that. Let's make it: "Two inflatable motorboats with outboard motors speed across the ocean, leaving white wakes, under a clear blue sky with a distinct horizon line, on a calm sea."

Now keywords. Need 10-18 unique, comma-separated. From context: UK, Europe, England, Seafront, Seaford (but the title hint says Seaford, England, UK, GBR, Europe). Wait, the title hint is "Seafront, Seaford, England, UK, GBR, Europe". But the image doesn't show a seafront, just the ocean. Wait, the descriptive hints say "Title hint: Seafront, Seaford, England, UK, GBR, Europe". But the image is of two boats on the ocean, so maybe Seaford is a location, but the image doesn't show it. Wait, the problem says: "Descriptive hints: Title hint
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
  the token limit; Keyword list has 166 terms (requested 10-18); Duplicate
  keywords: seaford, england, uk, europe, inflatable boat, motorboat, people,
  horizon, blue sky, clear, water, action, holiday, white wake, driver,
  yamaha, rigid hull, lady maverick, yamaha engine, clear day, open sea,
  summer, adventure, travel, tourism, seaside, gbr, blue, horizon line,
  inflatable, outboard motor, speed boat
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.45s
- *Generation time:* 66.91s
- *Total time:* 69.07s
- *Input validation time:* 0.69
- *Prompt preparation time:* 0.000579
- *First-token latency:* 45.7
- *Cleanup time:* 0.318
- *Prompt tokens:* 16,639
- *Generation tokens:* 500
- *Total tokens:* 17,139
- *Prompt throughput (raw):* 364 tok/s
- *Generation throughput (raw):* 25.3 tok/s
- *Peak memory:* 14
- *Active memory:* 9.5
- *Cache memory:* 1.3
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0103
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Model load time:* 3.97s
- *Generation time:* 5.64s
- *Total time:* 10.52s
- *Input validation time:* 0.867
- *Prompt preparation time:* 0.0257
- *First-token latency:* 0.37
- *Cleanup time:* 0.192
- *Prompt tokens:* 584
- *Generation tokens:* 81
- *Total tokens:* 665
- *Prompt throughput (raw):* 1,577 tok/s
- *Generation throughput (raw):* 17.7 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0103
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
&lt;channel|&gt;Title: Two Inflatable Boats Speeding Across Seaford Sea
Description: Two inflatable motorboats with outboard engines speed across the deep blue sea, leaving white wakes under a clear sky.
Keywords: Seaford, England, UK, ocean, inflatable boat, motorboat, water, water, action, people, horizon, sky, blue, vehicles, holiday, travel
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Two Inflatable Boats Speeding Across Seaford Sea
Description: Two inflatable motorboats with outboard engines speed across the deep blue sea, leaving white wakes under a clear sky.
Keywords: Seaford, England, UK, ocean, inflatable boat, motorboat, water, water, action, people, horizon, sky, blue, vehicles, holiday, travel
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
- *Model load time:* 3.97s
- *Generation time:* 5.61s
- *Total time:* 10.12s
- *Input validation time:* 0.52
- *Prompt preparation time:* 0.00047
- *First-token latency:* 0.379
- *Cleanup time:* 0.448
- *Prompt tokens:* 584
- *Generation tokens:* 82
- *Total tokens:* 666
- *Prompt throughput (raw):* 1,541 tok/s
- *Generation throughput (raw):* 19.4 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0103
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
&lt;channel|&gt;Title: Two Inflatable Boats Speeding Across the Seaford Sea
Description: Two inflatable boats with outboard motors speed across the blue ocean water, leaving white wakes behind them under a clear sky.
Keywords: Seaford, England, UK, ocean, inflatable boat, motorboat, people, action, water, blue sky, horizon, holiday, vehicles, riding, speed
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Two Inflatable Boats Speeding Across the Seaford Sea
Description: Two inflatable boats with outboard motors speed across the blue ocean water, leaving white wakes behind them under a clear sky.
Keywords: Seaford, England, UK, ocean, inflatable boat, motorboat, people, action, water, blue sky, horizon, holiday, vehicles, riding, speed
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
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava_next)
- *Model load time:* 0.92s
- *Generation time:* 6.86s
- *Total time:* 8.12s
- *Input validation time:* 0.335
- *Prompt preparation time:* 0.00162
- *First-token latency:* 5.06
- *Cleanup time:* 0.0893
- *Prompt tokens:* 2,305
- *Generation tokens:* 52
- *Total tokens:* 2,357
- *Prompt throughput (raw):* 456 tok/s
- *Generation throughput (raw):* 45.3 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.90
- *Model-load active memory:* 8.05
- *Post-cleanup active memory:* 0.0125
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Model load time:* 0.48s
- *Generation time:* 0.80s
- *Total time:* 1.60s
- *Input validation time:* 0.309
- *Prompt preparation time:* 0.00135
- *First-token latency:* 0.0824
- *Cleanup time:* 0.0805
- *Prompt tokens:* 315
- *Generation tokens:* 71
- *Total tokens:* 386
- *Prompt throughput (raw):* 3,821 tok/s
- *Generation throughput (raw):* 317 tok/s
- *Peak memory:* 2.2
- *Active memory:* 0.61
- *Cache memory:* 0.30
- *Model-load active memory:* 0.611
- *Post-cleanup active memory:* 0.0131
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: "Seaford in Action"
Description: A serene scene of two inflatable boats, one driving away from the other, captured in motion on a clear blue sky. The boats are driving on a calm sea, with a distinct horizon line in the background. The image is a testament to the beauty of nature and the joy of outdoor activities.
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
- *Generation time:* 21.49s
- *Total time:* 22.96s
- *Input validation time:* 0.306
- *Prompt preparation time:* 0.000172
- *First-token latency:* 2.06
- *Cleanup time:* 0.355
- *Prompt tokens:* 4,404
- *Generation tokens:* 500
- *Total tokens:* 4,904
- *Prompt throughput (raw):* 2,139 tok/s
- *Generation throughput (raw):* 27.1 tok/s
- *Peak memory:* 4.4
- *Active memory:* 1.7
- *Cache memory:* 1.9
- *Model-load active memory:* 1.73
- *Post-cleanup active memory:* 0.0142
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
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
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
- *Model load time:* 0.20s
- *Generation time:* -
- *Total time:* 0.59s
- *Input validation time:* 0.388
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0826
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.0027
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11841, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11318, in _load_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12271, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11856, in _run_model_generation
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
Fetching 54 files: 100%|##########| 54/54 [00:00<00:00, 3640.54it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[22:41:36] DEBUG    HF Cache Info for mlx-community/Inkling-Small-mlx-4bit: size=146358.0 MB, files=58
```

</details>

---
