# Model Output Gallery

Generated on: 2026-08-13 23:30:23 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|------------|-----------------|---------|------------|---------|-----------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | 1.15s   | 513 tok/s  | 0.05            | 1.2     | 358        | 82      | none                                                                                                |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 3.33s   | 184 tok/s  | 1.19            | 9.0     | 3,177      | 135     | none                                                                                                |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | 77.59s  | 63.7 tok/s | 63.29           | 74      | 16,482     | 124     | none                                                                                                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 63.98s  | 110 tok/s  | 59.66           | 24      | 16,482     | 120     | none                                                                                                |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 58.58s  | 93.5 tok/s | 55.31           | 10.0    | 16,482     | 109     | none                                                                                                |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable`              | 2.63s   | 126 tok/s  | 0.33            | 5.5     | 400        | 116     | none                                                                                                |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | 34.30s  | 43.8 tok/s | 20.96           | 70      | 3,468      | 117     | none                                                                                                |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 4.09s   | 130 tok/s  | 0.41            | 16      | 580        | 99      | none                                                                                                |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 8.48s   | 25.0 tok/s | 1.22            | 20      | 580        | 103     | none                                                                                                |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | 16.83s  | 94.5 tok/s | 13.94           | 8.4     | 16,467     | 139     | title/keyword constraints failed                                                                    |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | 10.20s  | 29.2 tok/s | 3.59            | 24      | 2,645      | 117     | title/keyword constraints failed                                                                    |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | 38.47s  | 33.7 tok/s | 23.66           | 78      | 6,310      | 141     | control tokens visible                                                                              |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | 10.71s  | 31.1 tok/s | 1.22            | 18      | 2,586      | 213     | role tokens visible                                                                                 |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable_with_caveats` | 5.92s   | 34.0 tok/s | 1.41            | 17      | 3,623      | 79      | title/keyword constraints failed                                                                    |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable_with_caveats` | 1.74s   | 195 tok/s  | 0.11            | 4.1     | 561        | 141     | title/keyword constraints failed                                                                    |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable_with_caveats` | 9.67s   | 19.1 tok/s | 1.82            | 15      | 275        | 113     | title/keyword constraints failed                                                                    |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable_with_caveats` | 2.11s   | 270 tok/s  | 0.27            | 3.8     | 964        | 107     | title/keyword constraints failed                                                                    |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable_with_caveats` | 6.94s   | 65.4 tok/s | 2.35            | 14      | 3,178      | 182     | title/keyword constraints failed                                                                    |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | 7.93s   | 60.7 tok/s | 2.64            | 15      | 3,178      | 207     | title/keyword constraints failed                                                                    |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | 5.26s   | 52.5 tok/s | 0.66            | 11      | 1,487      | 136     | title/keyword constraints failed                                                                    |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable_with_caveats` | 3.29s   | 56.0 tok/s | 0.29            | 9.6     | 1,094      | 94      | title/keyword constraints failed                                                                    |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable_with_caveats` | 71.63s  | 191 tok/s  | 69.91           | 5.1     | 16,478     | 63      | draft hints copied unchanged                                                                        |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | 23.05s  | 93.9 tok/s | 19.99           | 8.4     | 16,467     | 139     | title/keyword constraints failed                                                                    |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable_with_caveats` | 83.30s  | 17.8 tok/s | 71.78           | 35      | 16,482     | 131     | title/keyword constraints failed                                                                    |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | 5.68s   | 63.9 tok/s | 0.34            | 29      | 576        | 86      | control tokens visible; title/keyword constraints failed                                            |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | 5.21s   | 75.6 tok/s | 0.34            | 28      | 576        | 81      | control tokens visible                                                                              |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | 8.84s   | 27.9 tok/s | 1.13            | 18      | 569        | 139     | title/keyword constraints failed                                                                    |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | 6.50s   | 39.4 tok/s | 1.83            | 16      | 3,413      | 99      | title/keyword constraints failed                                                                    |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 16.00s  | 42.5 tok/s | 2.19            | 15      | 3,504      | 500     | missing required fields; echoes instructions; cut off at token limit                                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 28.43s  | 58.2 tok/s | 5.69            | 60      | 1,584      | 500     | missing required fields; extra text before Title; cut off at token limit                            |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 1.95s   | 346 tok/s  | 0.10            | 2.1     | 308        | 83      | missing required fields; echoes instructions                                                        |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 18.46s  | 46.9 tok/s | 5.71            | 13      | 6,306      | 500     | extra text before Title; cut off at token limit; incomplete thinking block                          |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `unusable`            | 14.33s  | 72.2 tok/s | 5.53            | 8.4     | 6,310      | 500     | repeated text; cut off at token limit; title/keyword constraints failed                             |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | 115.41s | 4.55 tok/s | 1.61            | 40      | 1,257      | 500     | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | 27.46s  | 5.98 tok/s | 2.53            | 24      | 3,100      | 131     | missing required fields                                                                             |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | 25.04s  | 90.2 tok/s | 17.99           | 8.4     | 16,469     | 500     | missing required fields; cut off at token limit; incomplete thinking block                          |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | 29.84s  | 53.7 tok/s | 18.58           | 13      | 16,478     | 500     | repeated text; cut off at token limit; title/keyword constraints failed                             |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 11.16s  | 48.7 tok/s | 0.38            | 17      | 568        | 391     | missing required fields; extra text before Title                                                    |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | 5.58s   | 62.2 tok/s | 2.40            | 9.7     | 2,680      | 102     | missing required fields                                                                             |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | 1.50s   | 353 tok/s  | 0.08            | 2.2     | 304        | 171     | missing required fields                                                                             |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | 15.26s  | 40.8 tok/s | 1.10            | 4.4     | 4,397      | 500     | repeated text; missing required fields; echoes instructions; cut off at token limit                 |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 513 tok/s

Average clean-completion throughput: 143 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.2 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Usability  | Observations                                                                                        |
|--------------------------------------------------------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------------------|
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable` | missing required fields; echoes instructions; cut off at token limit                                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable` | missing required fields; extra text before Title; cut off at token limit                            |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable` | missing required fields; echoes instructions                                                        |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                   | `unusable` | extra text before Title; cut off at token limit; incomplete thinking block                          |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                             | `unusable` | repeated text; cut off at token limit; title/keyword constraints failed                             |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | `unusable` | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                | `unusable` | missing required fields                                                                             |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                  | `unusable` | missing required fields; cut off at token limit; incomplete thinking block                          |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `unusable` | repeated text; cut off at token limit; title/keyword constraints failed                             |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                            | `unusable` | missing required fields; extra text before Title                                                    |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                 | `unusable` | missing required fields                                                                             |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable` | missing required fields                                                                             |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable` | repeated text; missing required fields; echoes instructions; cut off at token limit                 |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                                     |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | Title:<br>Seafront, Felixstowe, England, UK, GBR, Europe<br><br>Description:<br>A street lamp stands on a seaside promenade, with a backdrop of industrial cranes and a clear blue sky.<br><br>Keywords:<br>Seafront, Felixstowe, England, UK, GBR, Europe, street lamp, industrial cranes, seaside promen...                      |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>Felixstowe Seafront by the Port Cranes and Ferris Wheel<br><br>Description:<br>A coastal view at Felixstowe, UK, on 2026-08-13, captures a blend of industrial port activity with a leisurely Ferris wheel, under bright daylight with vintage-style street lamps and festive string li...                               |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | Title: Seafront view of Felixstowe pier and port cranes<br><br>Description: A sunlit seafront scene in Felixstowe, Suffolk, featuring a long pier extending over the water with a Ferris wheel visible in the distance. Large container port cranes dominate the hazy background skyline, w...                                     |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: Felixstowe Seafront with Pier and Port Cranes<br><br>Description: A sunlit view of the Felixstowe seafront featuring a pier extending into the water, with large port cranes and a Ferris wheel visible in the hazy background. Decorative string lights hang in the foreground alon...                                     |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | Title: Felixstowe Seafront with Port Cranes and Ferris Wheel<br><br>Description: A sunlit view of Felixstowe seafront, featuring vintage-style lampposts in the foreground, a pier with string lights, and distant port cranes and a Ferris wheel under a clear sky.<br><br>Keywords: Felixstowe,...                               |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable`              | Title:<br>Seafront, Felixstowe, England, UK, GBR, Europe<br>Description:<br>Capture date/time: 2026-08-13 16:08:57 UTC+01:00<br>GPS: 51.961800°N, 1.358500°E<br><br>Keywords:<br>Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront                                                    |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | Title:<br>Felixstowe Seafront with Port Cranes and Pier<br><br>Description:<br>A sunny seafront view in Felixstowe, England, featuring a Victorian lamp post in the foreground, a pier stretching across the water, a Ferris wheel, and large port cranes visible across the bay under bright day...                               |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: Seafront view of Felixstowe port and cranes<br>Description: A street lamp in the foreground frames a view of the Felixstowe seafront, featuring industrial port cranes and a pier under a clear sky. The scene captures the coastal landscape of East Suffolk with bright, natura...                                        |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: Seafront view of Felixstowe port and pier, Suffolk<br>Description: A view across the seafront in Felixstowe, Suffolk, featuring a pier and the industrial cranes of the port under a clear sky. A traditional lamp post stands in the foreground, framing the distant harbor and...                                         |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | Title:<br>Felixstowe Seafront at Sunset<br><br>Description:<br>A serene seafront view in Felixstowe, England, featuring a classic black lamppost in the foreground, colorful string lights strung across the water, and a distant pier with a Ferris wheel and large cranes. The scene is bathed...                                |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | Title:<br>Seafront View, Felixstowe, England<br><br>Description:<br>A serene seafront scene in Felixstowe, England, featuring a traditional lamp post in the foreground, a calm sea with a distant pier, and industrial cranes in the background under a clear sky.<br><br>Keywords:<br>Adobe Stock, Any V...                      |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: Felixstowe Seafront with Cranes and Pier<br>Description: A seafront scene in Felixstowe, England, featuring a pier, Ferris wheel, and industrial cranes in the distance, with lampposts, string lights, and a paddleboarder on the water under clear daylight.<br>Ke...                                   |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | Title: Felixstowe Seafront in Suffolk, England, UK.<br>Description: The image depicts a serene seafront in Felixstowe, Suffolk, England, UK, characterized by a calm body of water, likely the North Sea, stretching towards the horizon. In the background, a series of large, industri...                                        |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable_with_caveats` | Title: Felixstowe Seafront at Dusk<br>Description: The image captures a serene seafront in Felixstowe, England, with industrial cranes and a Ferris wheel in the background, under soft evening light.<br>Keywords: Felixstowe, seafront, England, Suffolk, East Suffolk, industrial cranes...                                     |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable_with_caveats` | Title: Felixstowe Seafront at Dusk<br><br>Description: A picturesque view of Felixstowe's seafront, showcasing the iconic East Suffolk pier and the silhouette of the famed Ferris wheel against a twilight sky. The scene is bathed in the warm glow of street lamps, with the calm waters...                                     |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable_with_caveats` | Title: A Seaside Scene in East Suffolk, England<br><br>Description: The image captures a serene seaside scene in East Suffolk, England, with a pier and a large industrial area visible in the background, set against a clear sky and calm waters, with a person paddleboarding in the dis...                                     |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable_with_caveats` | &lt;think&gt;<br><br>&lt;/think&gt;<br><br>Title:<br>Felixstowe Seafront with Industrial Cranes<br><br>Description:<br>A scenic seafront view with pier and ferris wheel under clear skies, showing industrial cranes and water.<br><br>Keywords: Felixstowe, seafront, England, UK, gbr, Europe, industrial, cranes, pier, fer... |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable_with_caveats` | Title:<br>**Felixstowe Seafront with Industrial Cranes and Ferris Wheel**<br><br>Description:<br>A serene seafront view of Felixstowe, England, captured on 13 August 2026, showcasing the tranquil water, distant industrial cranes, and a Ferris wheel under soft daylight. Decorative string l...                               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | **Title:**<br>*Felixstowe Seafront: Cranes, Pier, and Sunset Glow*<br><br>**Description:**<br>A serene coastal scene at Felixstowe, England, captured on **August 13, 2026, at 16:08 UTC+1**, showcasing industrial cranes along the docks, a historic pier, and a person stand-up paddleboarding...                               |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | Title: Seafront view of Felixstowe with illuminated pier and Ferris wheel<br><br>Description: A picturesque English seafront scene featuring a brightly lit pier, a Ferris wheel, and industrial cranes in the background. The sky is overcast, creating a moody atmosphere. A lone figure...                                      |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable_with_caveats` | Title: Felixstowe Seafront View<br><br>Description: A serene view of Felixstowe seafront with a backdrop of industrial cranes and a pier, captured on August 13, 2026, at 4:08 PM UTC+1.<br><br>Keywords: Felixstowe, seafront, industrial, cranes, pier, East Suffolk, England, UK, GBR, waterfront                               |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable_with_caveats` | Title: Seafront, Felixstowe, England, UK, GBR, Europe<br>Description: Seafront, Felixstowe, England, UK, GBR<br>Keywords: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront                                                                                                           |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | Title:<br>Felixstowe Seafront at Sunset<br><br>Description:<br>A serene seafront view in Felixstowe, England, featuring a classic black lamppost in the foreground, colorful string lights strung across the water, and a distant pier with a Ferris wheel and large cranes. The scene is bathed...                                |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable_with_caveats` | Title:<br>Felixstowe Seafront with Port Cranes and Pier<br><br>Description:<br>A view of the Felixstowe seafront in Suffolk, England, featuring a prominent pier and a Ferris wheel in the hazy distance. Foreground elements include a blurred street lamp and festive string lights, while larg...                               |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Seafront View and Industrial Port in Felixstowe Suffolk<br>Description: A view of the Felixstowe seafront featuring a pier, industrial cranes, and a Ferris wheel in the distance under a clear sky.<br>Keywords: Felixstowe, Suffolk, England, UK, seaf...                                |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Seafront view of Port of Felixstowe England<br>Description: A coastal view of the Felixstowe seafront featuring a pier, industrial cranes, and a Ferris wheel under a clear sky.<br>Keywords: Felixstowe, Suffolk, England, UK, seafront, harbor, port,...                                 |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | Title: Felixstowe Seafront: Pier, Cranes, and Lamp Posts<br><br>Description: Captured on 13th August 2026 at 16:08 UTC+01, this view shows the Felixstowe seafront with a pier, distant container cranes, and ornate lamp posts in the foreground. A calm sea and clear sky provide a brigh...                                     |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | Title:<br>Felixstowe Seafront with Lampposts and Cranes<br><br>Description:<br>The image captures a serene seafront in Felixstowe, England, featuring traditional lampposts and industrial cranes in the background under a clear sky.<br><br>Keywords:<br>Felixstowe, seafront, England, UK, lampposts, c...                      |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS. The descriptive hints are given. We need to treat capture date/time and GPS as authoritative facts, but not c...                                        |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. The hints are pretty clear: "Seafront, Felixstowe, England, UK, GBR, Europe." I'll go with "Seafront at Felixstowe, England, UK." It's concise and accurate.<br><br>Now...                                      |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | A tranquil seaside scene in Felixstowe, UK, captured on 2026-08-13 at 16:08:57 UTC+01:00, featuring a concrete 5-10-word title, a 1-2-sentence descriptive description, and 10-18 unique, comma-separated keywords covering relevant context and visible details.                                                                  |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. Looking at the image: seafront, Felixstowe, England, with cranes, pier, Ferris wheel. So maybe "Felixstowe Seafront with Industrial Cranes".<br><br>Then the description: need to combine context (date/time, lo...                               |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `unusable`            | Title: Felixstowe Seafront<br>Description: A view of the Felixstowe seafront, featuring a pier, industrial cranes, street lamps, and a tree-lined waterfront under bright daylight.<br>Keywords: Felixstowe seafront, England UK, pier, industrial cranes, street lamps, waterfront, bright...                                     |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | ◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.<br><br>First, the context says the capture date/time is 2026-08-13 16:08:57 UTC+01:00, and GP...                                     |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | Felixstowe Seafront: Ferris Wheel, Pier, and Industrial Port<br><br>A scenic Felixstowe seafront view captures a Ferris wheel, pier, and industrial port under a clear sky. The image shows a pier extending into the water, a Ferris wheel in the distance, and a port with cranes and bui...                                     |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | Got it, let's tackle this step by step. First, the title. The context says Felixstowe, England, UK, GBR. The image has a seafront with cranes, a Ferris wheel, street lamps. So a title should be specific. Maybe "Felixstowe Seafront, England, UK" but need 5-10 words. Let's see:...                                            |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | Title: Felixstowe Seafront, East Suffolk, UK<br><br>Description: A sunny view of Felixstowe's seafront, featuring a pier, Ferris wheel, cranes, and decorative string lights, with a lamp post in the foreground and a person paddleboarding in the distance.<br><br>Keywords: Felixstowe, East S...                               |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | ## Image Description<br><br>The image captures a scenic view of a waterfront area, likely a harbor or bay, on a bright day. The foreground is dominated by several black lampposts, with the closest one sharply in focus. The lampposts are ornate, featuring a classic lantern design.<br><br>B...                               |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | Seafront, Felixstowe, England, UK, GBR, Europe, 2026-08-13 16:08:57 UTC+01:00, 51.961800°N, 1.358500°E, Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront                                                                                                                             |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | Title: Seafront, Felixstowe, England, UK, GBR, Europe<br>Description: A 5-10-word title that describes the setting and context of the image, with a focus on the maritime theme. It also includes a 1-2-sentence description that combines relevant information about the location, such...                                        |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | - Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, GBR,<br>- Description hint: Seafront, Felixstowe, England, UK, GBR,<br>- Title hint: Seafront, Felixstowe, England, UK, GBR,<br>- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,<br>- Keyword hints...                               |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.14`
- `mlx`: `0.32.1.dev20260813+a8e24f202`
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
- *Date:* 2026-08-13 16:08:57 UTC+01:00
- *Time:* 16:08:57
- *GPS:* 51.961800°N, 1.358500°E

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
> &#45; Capture date/time: 2026-08-13 16:08:57 UTC+01:00
> &#45; GPS: 51.961800°N, 1.358500°E
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
- *Model load time:* 0.40s
- *Generation time:* 0.51s
- *Total time:* 1.15s
- *Input validation time:* 0.228
- *Prompt preparation time:* 0.00631
- *First-token latency:* 0.0516
- *Cleanup time:* 0.0616
- *Prompt tokens:* 358
- *Generation tokens:* 82
- *Total tokens:* 440
- *Prompt throughput (raw):* 6,933 tok/s
- *Generation throughput (raw):* 513 tok/s
- *Peak memory:* 1.2
- *Active memory:* 0.90
- *Cache memory:* 0.02
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1183
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
Seafront, Felixstowe, England, UK, GBR, Europe

Description:
A street lamp stands on a seaside promenade, with a backdrop of industrial cranes and a clear blue sky.

Keywords:
Seafront, Felixstowe, England, UK, GBR, Europe, street lamp, industrial cranes, seaside promenade, clear sky
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
- *Model load time:* 0.98s
- *Generation time:* 2.35s
- *Total time:* 3.33s
- *Input validation time:* 4.5e-05
- *Prompt preparation time:* 0.00686
- *First-token latency:* 1.19
- *Cleanup time:* 0.0838
- *Prompt tokens:* 3,177
- *Generation tokens:* 135
- *Total tokens:* 3,312
- *Prompt throughput (raw):* 2,661 tok/s
- *Generation throughput (raw):* 184 tok/s
- *Peak memory:* 9.0
- *Active memory:* 2.8
- *Cache memory:* 0.41
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00562
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3569
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
Felixstowe Seafront by the Port Cranes and Ferris Wheel

Description:
A coastal view at Felixstowe, UK, on 2026-08-13, captures a blend of industrial port activity with a leisurely Ferris wheel, under bright daylight with vintage-style street lamps and festive string lights along the rocky promenade.

Keywords:
Felixstowe, East Suffolk, UK seafront, UK GBR, Suffolk coast, industrial port, Ferris wheel, vintage street lamps, string lights, coastal promenade, industrial cranes, maritime activity, UK England, European seaside, historic lighting fixtures
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
- *Model load time:* 11.13s
- *Generation time:* 66.43s
- *Total time:* 77.59s
- *Input validation time:* 4.64e-05
- *Prompt preparation time:* 0.0145
- *First-token latency:* 63.3
- *Cleanup time:* 0.225
- *Prompt tokens:* 16,482
- *Generation tokens:* 124
- *Total tokens:* 16,606
- *Prompt throughput (raw):* 260 tok/s
- *Generation throughput (raw):* 63.7 tok/s
- *Peak memory:* 74
- *Active memory:* 70
- *Cache memory:* 0.88
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.00706
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1223
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
Title: Seafront view of Felixstowe pier and port cranes

Description: A sunlit seafront scene in Felixstowe, Suffolk, featuring a long pier extending over the water with a Ferris wheel visible in the distance. Large container port cranes dominate the hazy background skyline, while decorative string lights and traditional lamp posts frame the foreground.

Keywords: Felixstowe, Suffolk, England, UK, seafront, pier, Ferris wheel, port cranes, container terminal, lamp post, string lights, trees, water, summer, haze, coastal, East Anglia, GBR
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
- *Model load time:* 2.45s
- *Generation time:* 61.50s
- *Total time:* 63.98s
- *Input validation time:* 4.99e-05
- *Prompt preparation time:* 0.0149
- *First-token latency:* 59.7
- *Cleanup time:* 0.111
- *Prompt tokens:* 16,482
- *Generation tokens:* 120
- *Total tokens:* 16,602
- *Prompt throughput (raw):* 276 tok/s
- *Generation throughput (raw):* 110 tok/s
- *Peak memory:* 24
- *Active memory:* 21
- *Cache memory:* 0.88
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.00857
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1223
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
Title: Felixstowe Seafront with Pier and Port Cranes

Description: A sunlit view of the Felixstowe seafront featuring a pier extending into the water, with large port cranes and a Ferris wheel visible in the hazy background. Decorative string lights hang in the foreground alongside classic lampposts and greenery.

Keywords: Felixstowe, seafront, pier, port cranes, Ferris wheel, England, UK, Suffolk, East Suffolk, Europe, sea, water, lamppost, string lights, trees, rocks, summer, daylight
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
- *Model load time:* 1.36s
- *Generation time:* 57.22s
- *Total time:* 58.58s
- *Input validation time:* 4.94e-05
- *Prompt preparation time:* 0.000336
- *First-token latency:* 55.3
- *Cleanup time:* 0.0866
- *Prompt tokens:* 16,482
- *Generation tokens:* 109
- *Total tokens:* 16,591
- *Prompt throughput (raw):* 298 tok/s
- *Generation throughput (raw):* 93.5 tok/s
- *Peak memory:* 10.0
- *Active memory:* 6.6
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00909
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1223
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
Title: Felixstowe Seafront with Port Cranes and Ferris Wheel

Description: A sunlit view of Felixstowe seafront, featuring vintage-style lampposts in the foreground, a pier with string lights, and distant port cranes and a Ferris wheel under a clear sky.

Keywords: Felixstowe, seafront, port cranes, Ferris wheel, pier, lampposts, string lights, East Suffolk, England, UK, GBR, Europe, coastal view, harbour, summer day
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
- *Model load time:* 0.72s
- *Generation time:* 1.91s
- *Total time:* 2.63s
- *Input validation time:* 5.42e-05
- *Prompt preparation time:* 0.00186
- *First-token latency:* 0.329
- *Cleanup time:* 0.0757
- *Prompt tokens:* 400
- *Generation tokens:* 116
- *Total tokens:* 516
- *Prompt throughput (raw):* 1,216 tok/s
- *Generation throughput (raw):* 126 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.19
- *Model-load active memory:* 4.5
- *Post-cleanup active memory:* 0.00972
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1164
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
 Title:
Seafront, Felixstowe, England, UK, GBR, Europe
Description:
Capture date/time: 2026-08-13 16:08:57 UTC+01:00
GPS: 51.961800°N, 1.358500°E

Keywords:
Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront
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
- *Model load time:* 9.89s
- *Generation time:* 24.39s
- *Total time:* 34.30s
- *Input validation time:* 4.52e-05
- *Prompt preparation time:* 0.0159
- *First-token latency:* 21
- *Cleanup time:* 0.244
- *Prompt tokens:* 3,468
- *Generation tokens:* 117
- *Total tokens:* 3,585
- *Prompt throughput (raw):* 165 tok/s
- *Generation throughput (raw):* 43.8 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.42
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.00998
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1210
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
Felixstowe Seafront with Port Cranes and Pier

Description:
A sunny seafront view in Felixstowe, England, featuring a Victorian lamp post in the foreground, a pier stretching across the water, a Ferris wheel, and large port cranes visible across the bay under bright daylight.

Keywords:
Felixstowe, Suffolk, England, UK, seafront, pier, port cranes, Ferris wheel, Victorian lamp post, coastal view, East Suffolk, Europe, waterfront, maritime, industrial, sunny, blue water, scenic
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
- *Model load time:* 2.45s
- *Generation time:* 1.61s
- *Total time:* 4.09s
- *Input validation time:* 5.07e-05
- *Prompt preparation time:* 0.0205
- *First-token latency:* 0.413
- *Cleanup time:* 0.115
- *Prompt tokens:* 580
- *Generation tokens:* 99
- *Total tokens:* 679
- *Prompt throughput (raw):* 1,404 tok/s
- *Generation throughput (raw):* 130 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.20
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0119
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1186
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
Title: Seafront view of Felixstowe port and cranes
Description: A street lamp in the foreground frames a view of the Felixstowe seafront, featuring industrial port cranes and a pier under a clear sky. The scene captures the coastal landscape of East Suffolk with bright, natural daylight.
Keywords: Felixstowe, Suffolk, England, UK, seafront, port, cranes, pier, coast, maritime, industrial, Europe, East Suffolk, street lamp, coastal, water, daytime
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
- *Model load time:* 2.69s
- *Generation time:* 5.78s
- *Total time:* 8.48s
- *Input validation time:* 4.77e-05
- *Prompt preparation time:* 0.000309
- *First-token latency:* 1.22
- *Cleanup time:* 0.134
- *Prompt tokens:* 580
- *Generation tokens:* 103
- *Total tokens:* 683
- *Prompt throughput (raw):* 475 tok/s
- *Generation throughput (raw):* 25.0 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.76
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0124
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1186
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
Title: Seafront view of Felixstowe port and pier, Suffolk
Description: A view across the seafront in Felixstowe, Suffolk, featuring a pier and the industrial cranes of the port under a clear sky. A traditional lamp post stands in the foreground, framing the distant harbor and Ferris wheel.
Keywords: Felixstowe, Suffolk, England, UK, seafront, port, harbor, cranes, pier, Ferris wheel, lamp post, industrial, coast, East Suffolk, maritime, Europe, summer
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
- *Observations:* Title has 4 words (requested 5-10); Keyword list has 25
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.67s
- *Generation time:* 16.14s
- *Total time:* 16.83s
- *Input validation time:* 4.74e-05
- *Prompt preparation time:* 0.0089
- *First-token latency:* 13.9
- *Cleanup time:* 0.0841
- *Prompt tokens:* 16,467
- *Generation tokens:* 139
- *Total tokens:* 16,606
- *Prompt throughput (raw):* 1,182 tok/s
- *Generation throughput (raw):* 94.5 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1204
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
Felixstowe Seafront at Sunset

Description:
A serene seafront view in Felixstowe, England, featuring a classic black lamppost in the foreground, colorful string lights strung across the water, and a distant pier with a Ferris wheel and large cranes. The scene is bathed in soft, golden light, suggesting late afternoon or early evening.

Keywords: Felixstowe, seafront, England, UK, Europe, Suffolk, gbr, coastal, pier, Ferris wheel, cranes, lamppost, sunset, water, sky, trees, light, beach, harbor, maritime, coastal town, seaside, British, landscape, view
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
- *Observations:* Title has 4 words (requested 5-10); Keyword list has 21
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.15s
- *Generation time:* 8.04s
- *Total time:* 10.20s
- *Input validation time:* 5.23e-05
- *Prompt preparation time:* 0.00679
- *First-token latency:* 3.59
- *Cleanup time:* 0.117
- *Prompt tokens:* 2,645
- *Generation tokens:* 117
- *Total tokens:* 2,762
- *Prompt throughput (raw):* 737 tok/s
- *Generation throughput (raw):* 29.2 tok/s
- *Peak memory:* 24
- *Active memory:* 18
- *Cache memory:* 0.52
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.000968
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1132
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
Seafront View, Felixstowe, England

Description:
A serene seafront scene in Felixstowe, England, featuring a traditional lamp post in the foreground, a calm sea with a distant pier, and industrial cranes in the background under a clear sky.

Keywords:
Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, GBR, seafront, lamp post, sea, pier, industrial cranes, clear sky, serene, coastal view, traditional, waterfront, summer, daylight
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
- *Observations:* Unrecognised model control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 10.09s
- *Generation time:* 28.36s
- *Total time:* 38.47s
- *Input validation time:* 4.61e-05
- *Prompt preparation time:* 0.00762
- *First-token latency:* 23.7
- *Cleanup time:* 0.35
- *Prompt tokens:* 6,310
- *Generation tokens:* 141
- *Total tokens:* 6,451
- *Prompt throughput (raw):* 267 tok/s
- *Generation throughput (raw):* 33.7 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00243
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1213
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
&lt;|begin_of_box|&gt;Title: Felixstowe Seafront with Cranes and Pier
Description: A seafront scene in Felixstowe, England, featuring a pier, Ferris wheel, and industrial cranes in the distance, with lampposts, string lights, and a paddleboarder on the water under clear daylight.
Keywords: Felixstowe, seafront, England, UK, GBR, pier, Ferris wheel, industrial cranes, lampposts, string lights, paddleboarder, East Suffolk, Europe, 2026-08-13, 16:08:57, 51.961800°N, 1.358500°E&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Felixstowe Seafront with Cranes and Pier
Description: A seafront scene in Felixstowe, England, featuring a pier, Ferris wheel, and industrial cranes in the distance, with lampposts, string lights, and a paddleboarder on the water under clear daylight.
Keywords: Felixstowe, seafront, England, UK, GBR, pier, Ferris wheel, industrial cranes, lampposts, string lights, paddleboarder, East Suffolk, Europe, 2026-08-13, 16:08:57, 51.961800°N, 1.358500°E<|end_of_box|>
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
- *Model load time:* 2.02s
- *Generation time:* 8.68s
- *Total time:* 10.71s
- *Input validation time:* 7.75e-05
- *Prompt preparation time:* 0.00134
- *First-token latency:* 1.22
- *Cleanup time:* 0.24
- *Prompt tokens:* 2,586
- *Generation tokens:* 213
- *Total tokens:* 2,799
- *Prompt throughput (raw):* 2,126 tok/s
- *Generation throughput (raw):* 31.1 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.00269
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1169
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
 Title: Felixstowe Seafront in Suffolk, England, UK.
Description: The image depicts a serene seafront in Felixstowe, Suffolk, England, UK, characterized by a calm body of water, likely the North Sea, stretching towards the horizon. In the background, a series of large, industrial cranes are visible, indicating a port or shipping area. A pier extends into the water, adorned with decorative lights that create a festive atmosphere. The sky is clear, suggesting a sunny day. The foreground features a row of street lamps, adding to the ambiance of the scene. Trees and greenery are present, providing a natural contrast to the man-made structures. The overall mood of the image is tranquil and picturesque, capturing the beauty of a coastal town with its industrial and natural elements harmoniously coexisting.
Keywords: Felixstowe, Suffolk, England, UK, seafront, North Sea, cranes, pier, decorative lights, street lamps, trees, greenery, coastal town, industrial, natural, serene, picturesque, tranquil.&lt;end_of_utterance&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Title: Felixstowe Seafront in Suffolk, England, UK.
Description: The image depicts a serene seafront in Felixstowe, Suffolk, England, UK, characterized by a calm body of water, likely the North Sea, stretching towards the horizon. In the background, a series of large, industrial cranes are visible, indicating a port or shipping area. A pier extends into the water, adorned with decorative lights that create a festive atmosphere. The sky is clear, suggesting a sunny day. The foreground features a row of street lamps, adding to the ambiance of the scene. Trees and greenery are present, providing a natural contrast to the man-made structures. The overall mood of the image is tranquil and picturesque, capturing the beauty of a coastal town with its industrial and natural elements harmoniously coexisting.
Keywords: Felixstowe, Suffolk, England, UK, seafront, North Sea, cranes, pier, decorative lights, street lamps, trees, greenery, coastal town, industrial, natural, serene, picturesque, tranquil.<end_of_utterance>
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
- *Model load time:* 1.67s
- *Generation time:* 4.24s
- *Total time:* 5.92s
- *Input validation time:* 7.5e-05
- *Prompt preparation time:* 0.00176
- *First-token latency:* 1.41
- *Cleanup time:* 0.111
- *Prompt tokens:* 3,623
- *Generation tokens:* 79
- *Total tokens:* 3,702
- *Prompt throughput (raw):* 2,568 tok/s
- *Generation throughput (raw):* 34.0 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.28
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.003
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1137
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
Title: Felixstowe Seafront at Dusk
Description: The image captures a serene seafront in Felixstowe, England, with industrial cranes and a Ferris wheel in the background, under soft evening light.
Keywords: Felixstowe, seafront, England, Suffolk, East Suffolk, industrial cranes, Ferris wheel, evening light, sea, GBR, Europe
</pre>

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### mlx-community/LFM2.5-VL-1.6B-bf16

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10); Keyword list has 19
  terms (requested 10-18); Duplicate keywords: felixstowe, seafront, uk, gbr
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.58s
- *Generation time:* 1.15s
- *Total time:* 1.74s
- *Input validation time:* 3.42e-05
- *Prompt preparation time:* 0.00338
- *First-token latency:* 0.109
- *Cleanup time:* 0.0765
- *Prompt tokens:* 561
- *Generation tokens:* 141
- *Total tokens:* 702
- *Prompt throughput (raw):* 5,152 tok/s
- *Generation throughput (raw):* 195 tok/s
- *Peak memory:* 4.1
- *Active memory:* 3.2
- *Cache memory:* 0.02
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.00379
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1183
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
Title: Felixstowe Seafront at Dusk

Description: A picturesque view of Felixstowe's seafront, showcasing the iconic East Suffolk pier and the silhouette of the famed Ferris wheel against a twilight sky. The scene is bathed in the warm glow of street lamps, with the calm waters of the English Channel reflecting the serene atmosphere.

Keywords: Felixstowe, Seafront, East Suffolk, England, UK, GBR, Seafront, Felixstowe, Suffolk, UK, gbr, seafront, Ferris wheel, twilight, street lamps, calm waters, English Channel, warm glow, serene atmosphere
</pre>

</details>

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

<details>
<summary>Complete evidence: mlx-community/Llama-3.2-11B-Vision-Instruct-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 21 terms (requested 10-18); Duplicate
  keywords: trees, water, sky, industrial cranes, buildings
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 1.51s
- *Generation time:* 8.15s
- *Total time:* 9.67s
- *Input validation time:* 4.57e-05
- *Prompt preparation time:* 0.00133
- *First-token latency:* 1.82
- *Cleanup time:* 0.116
- *Prompt tokens:* 275
- *Generation tokens:* 113
- *Total tokens:* 388
- *Prompt throughput (raw):* 151 tok/s
- *Generation throughput (raw):* 19.1 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.59
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.00431
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1236
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
Title: A Seaside Scene in East Suffolk, England

Description: The image captures a serene seaside scene in East Suffolk, England, with a pier and a large industrial area visible in the background, set against a clear sky and calm waters, with a person paddleboarding in the distance.

Keywords: East Suffolk, England, seaside, pier, industrial area, clear sky, calm waters, paddleboarding, person, trees, street lamps, rocks, water, sky, industrial cranes, buildings, trees, water, sky, industrial cranes, buildings.
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Model load time:* 0.90s
- *Generation time:* 1.19s
- *Total time:* 2.11s
- *Input validation time:* 5.01e-05
- *Prompt preparation time:* 0.00815
- *First-token latency:* 0.27
- *Cleanup time:* 0.089
- *Prompt tokens:* 964
- *Generation tokens:* 107
- *Total tokens:* 1,071
- *Prompt throughput (raw):* 3,574 tok/s
- *Generation throughput (raw):* 270 tok/s
- *Peak memory:* 3.8
- *Active memory:* 2.3
- *Cache memory:* 0.06
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00483
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1200
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

&lt;/think&gt;

Title:
Felixstowe Seafront with Industrial Cranes

Description:
A scenic seafront view with pier and ferris wheel under clear skies, showing industrial cranes and water.

Keywords: Felixstowe, seafront, England, UK, gbr, Europe, industrial, cranes, pier, ferriswheel, water, daylight, trees, streetlamps, decorations, rocky, blue, clear, 2026, 16:08:57
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>

</think>

Title:
Felixstowe Seafront with Industrial Cranes

Description:
A scenic seafront view with pier and ferris wheel under clear skies, showing industrial cranes and water.

Keywords: Felixstowe, seafront, England, UK, gbr, Europe, industrial, cranes, pier, ferriswheel, water, daylight, trees, streetlamps, decorations, rocky, blue, clear, 2026, 16:08:57
```

</details>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-mxfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.39s
- *Generation time:* 5.55s
- *Total time:* 6.94s
- *Input validation time:* 3.59e-05
- *Prompt preparation time:* 0.00669
- *First-token latency:* 2.35
- *Cleanup time:* 0.0936
- *Prompt tokens:* 3,178
- *Generation tokens:* 182
- *Total tokens:* 3,360
- *Prompt throughput (raw):* 1,354 tok/s
- *Generation throughput (raw):* 65.4 tok/s
- *Peak memory:* 14
- *Active memory:* 8.0
- *Cache memory:* 0.65
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.0051
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3570
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
**Felixstowe Seafront with Industrial Cranes and Ferris Wheel**

Description:
A serene seafront view of Felixstowe, England, captured on 13 August 2026, showcasing the tranquil water, distant industrial cranes, and a Ferris wheel under soft daylight. Decorative string lights and vintage-style lampposts frame the scene, with a lone paddleboarder adding subtle activity.

Keywords:
Felixstowe, seafront, England, UK, GBR, Europe, East Suffolk, industrial cranes, Ferris wheel, coastal view, paddleboarding, vintage lampposts, string lights, waterfront, 2026-08-13, 51.961800°N 1.358500°E, daylight, maritime, leisure, infrastructure
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
- *Observations:* Keyword list has 21 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.46s
- *Generation time:* 6.47s
- *Total time:* 7.93s
- *Input validation time:* 3.5e-05
- *Prompt preparation time:* 0.00025
- *First-token latency:* 2.64
- *Cleanup time:* 0.103
- *Prompt tokens:* 3,178
- *Generation tokens:* 207
- *Total tokens:* 3,385
- *Prompt throughput (raw):* 1,203 tok/s
- *Generation throughput (raw):* 60.7 tok/s
- *Peak memory:* 15
- *Active memory:* 8.4
- *Cache memory:* 0.65
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00536
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3570
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
*Felixstowe Seafront: Cranes, Pier, and Sunset Glow*

**Description:**
A serene coastal scene at Felixstowe, England, captured on **August 13, 2026, at 16:08 UTC+1**, showcasing industrial cranes along the docks, a historic pier, and a person stand-up paddleboarding. Decorative string lights and vintage-style lanterns frame the waterfront, bathed in soft evening sunlight.

**Keywords:**
Felixstowe, England, UK, GBR, Europe, East Suffolk, seafront, industrial cranes, dockyard, pier, stand-up paddleboarding, sunset, coastal lighting, vintage lanterns, decorative string lights, waterfront, historic docks, 2026-08-13, 16:08 UTC+1, 51.961800°N, 1.358500°E
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
- *Observations:* Keyword list has 21 terms (requested 10-18); Duplicate
  keywords: felixstowe, england, uk, gbr, europe
- *Arch supported by installed mlx-vlm:* yes (model_type molmo)
- *Model load time:* 1.17s
- *Generation time:* 4.09s
- *Total time:* 5.26s
- *Input validation time:* 4.04e-05
- *Prompt preparation time:* 0.00142
- *First-token latency:* 0.655
- *Cleanup time:* 0.0882
- *Prompt tokens:* 1,487
- *Generation tokens:* 136
- *Total tokens:* 1,623
- *Prompt throughput (raw):* 2,269 tok/s
- *Generation throughput (raw):* 52.5 tok/s
- *Peak memory:* 11
- *Active memory:* 9.0
- *Cache memory:* 0.14
- *Model-load active memory:* 9.04
- *Post-cleanup active memory:* 0.00593
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1128
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
 Title: Seafront view of Felixstowe with illuminated pier and Ferris wheel

Description: A picturesque English seafront scene featuring a brightly lit pier, a Ferris wheel, and industrial cranes in the background. The sky is overcast, creating a moody atmosphere. A lone figure stands on the pier, adding a sense of scale to the expansive view.

Keywords: Seafront, Felixstowe, England, UK, GBR, Europe, Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, pier, Ferris wheel, industrial cranes, overcast sky, lone figure, expansive view
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
- *Model load time:* 0.96s
- *Generation time:* 2.33s
- *Total time:* 3.29s
- *Input validation time:* 5.15e-05
- *Prompt preparation time:* 0.000892
- *First-token latency:* 0.291
- *Cleanup time:* 0.0788
- *Prompt tokens:* 1,094
- *Generation tokens:* 94
- *Total tokens:* 1,188
- *Prompt throughput (raw):* 3,753 tok/s
- *Generation throughput (raw):* 56.0 tok/s
- *Peak memory:* 9.6
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00713
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1153
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
Title: Felixstowe Seafront View

Description: A serene view of Felixstowe seafront with a backdrop of industrial cranes and a pier, captured on August 13, 2026, at 4:08 PM UTC+1.

Keywords: Felixstowe, seafront, industrial, cranes, pier, East Suffolk, England, UK, GBR, waterfront
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
- *Observations:* Title, Description and Keywords copy all supplied hints
  unchanged
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.66s
- *Generation time:* 70.96s
- *Total time:* 71.63s
- *Input validation time:* 4.44e-05
- *Prompt preparation time:* 0.00336
- *First-token latency:* 69.9
- *Cleanup time:* 0.106
- *Prompt tokens:* 16,478
- *Generation tokens:* 63
- *Total tokens:* 16,541
- *Prompt throughput (raw):* 236 tok/s
- *Generation throughput (raw):* 191 tok/s
- *Peak memory:* 5.1
- *Active memory:* 1.3
- *Cache memory:* 0.85
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.00744
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1262
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
Title: Seafront, Felixstowe, England, UK, GBR, Europe
Description: Seafront, Felixstowe, England, UK, GBR
Keywords: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront
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
- *Observations:* Title has 4 words (requested 5-10); Keyword list has 25
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.73s
- *Generation time:* 22.32s
- *Total time:* 23.05s
- *Input validation time:* 4.96e-05
- *Prompt preparation time:* 0.00023
- *First-token latency:* 20
- *Cleanup time:* 0.0871
- *Prompt tokens:* 16,467
- *Generation tokens:* 139
- *Total tokens:* 16,606
- *Prompt throughput (raw):* 824 tok/s
- *Generation throughput (raw):* 93.9 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00775
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1204
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
Felixstowe Seafront at Sunset

Description:
A serene seafront view in Felixstowe, England, featuring a classic black lamppost in the foreground, colorful string lights strung across the water, and a distant pier with a Ferris wheel and large cranes. The scene is bathed in soft, golden light, suggesting late afternoon or early evening.

Keywords: Felixstowe, seafront, England, UK, Europe, Suffolk, gbr, coastal, pier, Ferris wheel, cranes, lamppost, sunset, water, sky, trees, light, beach, harbor, maritime, coastal town, seaside, British, landscape, view
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
- *Model load time:* 3.20s
- *Generation time:* 80.08s
- *Total time:* 83.30s
- *Input validation time:* 4.81e-05
- *Prompt preparation time:* 0.0156
- *First-token latency:* 71.8
- *Cleanup time:* 0.141
- *Prompt tokens:* 16,482
- *Generation tokens:* 131
- *Total tokens:* 16,613
- *Prompt throughput (raw):* 230 tok/s
- *Generation throughput (raw):* 17.8 tok/s
- *Peak memory:* 35
- *Active memory:* 29
- *Cache memory:* 1.8
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.0096
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1223
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
Felixstowe Seafront with Port Cranes and Pier

Description:
A view of the Felixstowe seafront in Suffolk, England, featuring a prominent pier and a Ferris wheel in the hazy distance. Foreground elements include a blurred street lamp and festive string lights, while large industrial port cranes loom over the coastline under a bright, sunny sky.

Keywords:
Felixstowe, Suffolk, England, UK, Europe, seafront, port, cranes, pier, Ferris wheel, street lamp, string lights, ocean, coastline, summer, hazy, industrial, landscape, travel
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
  keywords: travel
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.36s
- *Generation time:* 2.30s
- *Total time:* 5.68s
- *Input validation time:* 4.79e-05
- *Prompt preparation time:* 0.0206
- *First-token latency:* 0.337
- *Cleanup time:* 0.127
- *Prompt tokens:* 576
- *Generation tokens:* 86
- *Total tokens:* 662
- *Prompt throughput (raw):* 1,710 tok/s
- *Generation throughput (raw):* 63.9 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1158
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
&lt;channel|&gt;Title: Seafront View and Industrial Port in Felixstowe Suffolk
Description: A view of the Felixstowe seafront featuring a pier, industrial cranes, and a Ferris wheel in the distance under a clear sky.
Keywords: Felixstowe, Suffolk, England, UK, seafront, port, cranes, pier, Ferris wheel, coastal, maritime, water, East Suffolk, industrial, travel, travel
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Seafront View and Industrial Port in Felixstowe Suffolk
Description: A view of the Felixstowe seafront featuring a pier, industrial cranes, and a Ferris wheel in the distance under a clear sky.
Keywords: Felixstowe, Suffolk, England, UK, seafront, port, cranes, pier, Ferris wheel, coastal, maritime, water, East Suffolk, industrial, travel, travel
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
- *Model load time:* 3.19s
- *Generation time:* 2.02s
- *Total time:* 5.21s
- *Input validation time:* 4.43e-05
- *Prompt preparation time:* 0.000303
- *First-token latency:* 0.337
- *Cleanup time:* 0.151
- *Prompt tokens:* 576
- *Generation tokens:* 81
- *Total tokens:* 657
- *Prompt throughput (raw):* 1,709 tok/s
- *Generation throughput (raw):* 75.6 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1158
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
&lt;channel|&gt;Title: Seafront view of Port of Felixstowe England
Description: A coastal view of the Felixstowe seafront featuring a pier, industrial cranes, and a Ferris wheel under a clear sky.
Keywords: Felixstowe, Suffolk, England, UK, seafront, harbor, port, cranes, pier, Ferris wheel, industrial, coastal, lamp, East Suffolk, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Seafront view of Port of Felixstowe England
Description: A coastal view of the Felixstowe seafront featuring a pier, industrial cranes, and a Ferris wheel under a clear sky.
Keywords: Felixstowe, Suffolk, England, UK, seafront, harbor, port, cranes, pier, Ferris wheel, industrial, coastal, lamp, East Suffolk, Europe
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
- *Model load time:* 2.32s
- *Generation time:* 6.52s
- *Total time:* 8.84s
- *Input validation time:* 5.02e-05
- *Prompt preparation time:* 0.00314
- *First-token latency:* 1.13
- *Cleanup time:* 0.128
- *Prompt tokens:* 569
- *Generation tokens:* 139
- *Total tokens:* 708
- *Prompt throughput (raw):* 502 tok/s
- *Generation throughput (raw):* 27.9 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.43
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0108
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1187
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
Title: Felixstowe Seafront: Pier, Cranes, and Lamp Posts

Description: Captured on 13th August 2026 at 16:08 UTC+01, this view shows the Felixstowe seafront with a pier, distant container cranes, and ornate lamp posts in the foreground. A calm sea and clear sky provide a bright backdrop to the coastal scene.

Keywords: Felixstowe, Suffolk, England, UK, GBR, East Suffolk, Seafront, Pier, Container Cranes, Lamp Post, Coastal, Beach, Summer, Daylight, Blue Sky, Port, Maritime, 2026, UTC+01
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
- *Observations:* Duplicate keywords: seafront
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.74s
- *Generation time:* 4.76s
- *Total time:* 6.50s
- *Input validation time:* 0.000124
- *Prompt preparation time:* 0.00437
- *First-token latency:* 1.83
- *Cleanup time:* 0.0932
- *Prompt tokens:* 3,413
- *Generation tokens:* 99
- *Total tokens:* 3,512
- *Prompt throughput (raw):* 1,864 tok/s
- *Generation throughput (raw):* 39.4 tok/s
- *Peak memory:* 16
- *Active memory:* 13
- *Cache memory:* 0.64
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0144
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1132
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
Felixstowe Seafront with Lampposts and Cranes

Description:
The image captures a serene seafront in Felixstowe, England, featuring traditional lampposts and industrial cranes in the background under a clear sky.

Keywords:
Felixstowe, seafront, England, UK, lampposts, cranes, waterfront, industrial, Europe, Suffolk, East Suffolk, seafront, Adobe Stock, Any Vision
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
- *Model load time:* 1.61s
- *Generation time:* 14.38s
- *Total time:* 16.00s
- *Input validation time:* 5.57e-05
- *Prompt preparation time:* 0.0067
- *First-token latency:* 2.19
- *Cleanup time:* 0.0863
- *Prompt tokens:* 3,504
- *Generation tokens:* 500
- *Total tokens:* 4,004
- *Prompt throughput (raw):* 1,600 tok/s
- *Generation throughput (raw):* 42.5 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.82
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.000706
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1506
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
We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS. The descriptive hints are given. We need to treat capture date/time and GPS as authoritative facts, but not claim they are visible. So we can incorporate them into description if needed (like "captured on 13 August 2026 at 16:08:57 UTC+01:00, at 51.961800°N, 1.358500°E").

We need to produce:

- a concrete 5-10-word title (so between 5 and 10 words inclusive). Must be concrete, not hint. Should reflect the main subject: seafront with lampposts, ferry, cranes, Ferris wheel, etc. Title: "Felixstowe Seafront at Sunset with Lampposts". That's 7 words: Felixstowe (1) Seafront (2) at (3) Sunset (4) with (5) Lampposts (6). That's 6 words. Could add "and" maybe "and Ferris wheel". "Felixstowe Seafront at Sunset with Lampposts and Ferris Wheel". That's 10 words? Let's count: Felixstowe(1) Seafront(2) at(3) Sunset(4) with(5) Lampposts(6) and(7) Ferris(8) Wheel(9). That's 9 words. Good. Or "Felixstowe Seafront Sunset Lampposts and Cranes". Count: Felixstowe(1) Seafront(2) Sunset(3) Lampposts(4) and(5) Cranes(6). That's 6 words. But we need 5-10 words, so any is fine.

- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details. Should be 1-2 sentences. Should mention location (Felixstowe, East Suffolk, England, UK), seafront, lampposts, Ferris wheel, cargo cranes, water, string lights, etc. Also mention time of day: appears to be late
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
- *Observations:* Missing or empty fields: Description, Keywords; Extra text
  appears before the Title field; Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 12.21s
- *Generation time:* 16.20s
- *Total time:* 28.43s
- *Input validation time:* 6.57e-05
- *Prompt preparation time:* 0.0102
- *First-token latency:* 5.69
- *Cleanup time:* 0.228
- *Prompt tokens:* 1,584
- *Generation tokens:* 500
- *Total tokens:* 2,084
- *Prompt throughput (raw):* 279 tok/s
- *Generation throughput (raw):* 58.2 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.15
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00118
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1322
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
Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. The hints are pretty clear: "Seafront, Felixstowe, England, UK, GBR, Europe." I'll go with "Seafront at Felixstowe, England, UK." It's concise and accurate.

Now for the description. I need to weave in the context – the date, time, and GPS – with the visual elements. The image is a twilight scene, so I'll mention that. The main subject is the seafront, with the lighthouse and the ferries. The hints mention the ferries, so I'll include that. The lighting is key, so I'll describe the sunset and the warm glow. I'll also mention the ferries and the lighthouse.

Here's my draft: "A twilight view of the Felixstowe seafront at 4:08 PM on August 13, 2026. The scene features a lighthouse against a sunset sky, with ferries and a distant view of the North Sea." That covers the essentials.

Finally, the keywords. I need to be comprehensive. I'll start with the location, then the time of day, the subject matter, and the key objects. I'll also include the style and the mood. I'll make sure to include the authoritative context and the hints.

Here's my final list: Felixstowe, seafront, England, UK, GBR, Europe, twilight, lighthouse, ferries, North Sea, sunset, 2026-08-13, 16:08:57, 51.961800, 1.358500, East Suffolk, Suffolk, Adobe Stock, Any Vision, ferry, lighthouse, North Sea, sunset, twilight, seafront, England, UK, GBR, Europe, Felixstowe, Suffolk, UK, gbr, seafront.

I've checked everything. The title is good, the description is solid, and the keywords are comprehensive. I'm ready to output the final metadata.
&lt;/think&gt;

Title: Seafront at Felix
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. The hints are pretty clear: "Seafront, Felixstowe, England, UK, GBR, Europe." I'll go with "Seafront at Felixstowe, England, UK." It's concise and accurate.

Now for the description. I need to weave in the context – the date, time, and GPS – with the visual elements. The image is a twilight scene, so I'll mention that. The main subject is the seafront, with the lighthouse and the ferries. The hints mention the ferries, so I'll include that. The lighting is key, so I'll describe the sunset and the warm glow. I'll also mention the ferries and the lighthouse.

Here's my draft: "A twilight view of the Felixstowe seafront at 4:08 PM on August 13, 2026. The scene features a lighthouse against a sunset sky, with ferries and a distant view of the North Sea." That covers the essentials.

Finally, the keywords. I need to be comprehensive. I'll start with the location, then the time of day, the subject matter, and the key objects. I'll also include the style and the mood. I'll make sure to include the authoritative context and the hints.

Here's my final list: Felixstowe, seafront, England, UK, GBR, Europe, twilight, lighthouse, ferries, North Sea, sunset, 2026-08-13, 16:08:57, 51.961800, 1.358500, East Suffolk, Suffolk, Adobe Stock, Any Vision, ferry, lighthouse, North Sea, sunset, twilight, seafront, England, UK, GBR, Europe, Felixstowe, Suffolk, UK, gbr, seafront.

I've checked everything. The title is good, the description is solid, and the keywords are comprehensive. I'm ready to output the final metadata.
</think>

Title: Seafront at Felix
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
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response repeats the task instructions instead of only returning the
  requested fields
- *Arch supported by installed mlx-vlm:* yes (model_type llava_qwen2 via
  fastvlm)
- *Model load time:* 0.62s
- *Generation time:* 1.32s
- *Total time:* 1.95s
- *Input validation time:* 6.38e-05
- *Prompt preparation time:* 0.00169
- *First-token latency:* 0.0994
- *Cleanup time:* 0.0712
- *Prompt tokens:* 308
- *Generation tokens:* 83
- *Total tokens:* 391
- *Prompt throughput (raw):* 3,099 tok/s
- *Generation throughput (raw):* 346 tok/s
- *Peak memory:* 2.1
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.00149
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1227
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
A tranquil seaside scene in Felixstowe, UK, captured on 2026-08-13 at 16:08:57 UTC+01:00, featuring a concrete 5-10-word title, a 1-2-sentence descriptive description, and 10-18 unique, comma-separated keywords covering relevant context and visible details.
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
  cut off at the token limit; Internal reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.70s
- *Generation time:* 16.76s
- *Total time:* 18.46s
- *Input validation time:* 4.75e-05
- *Prompt preparation time:* 0.00166
- *First-token latency:* 5.71
- *Cleanup time:* 0.0898
- *Prompt tokens:* 6,306
- *Generation tokens:* 500
- *Total tokens:* 6,806
- *Prompt throughput (raw):* 1,103 tok/s
- *Generation throughput (raw):* 46.9 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.41
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1189
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
&lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. Looking at the image: seafront, Felixstowe, England, with cranes, pier, Ferris wheel. So maybe "Felixstowe Seafront with Industrial Cranes".

Then the description: need to combine context (date/time, location) with main subjects (seafront, cranes, pier, Ferris wheel, lighting, action). The image has a seafront with industrial cranes, a pier, Ferris wheel, street lamps, string lights, water, and a person maybe. So something like "Seafront view of Felixstowe, England, showing industrial cranes, pier, Ferris wheel, and waterfront with street lamps under bright daylight." Wait, need to include the context like capture date, but maybe not in the description, but the description should mention the setting. Let's make it: "Seafront scene in Felixstowe, England, featuring industrial cranes, a pier, Ferris wheel, and waterfront with street lamps under bright daylight conditions."

Keywords need 10-18, unique, comma-separated. From hints: East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront, Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. Wait, need unique ones. Let's list: Felixstowe, Suffolk, UK, seafront, industrial cranes, pier, Ferris wheel, street lamps, waterfront, Europe, England. Wait, let's check the image: street lamps, string lights, water, cranes, pier, Ferris wheel, trees, rocks. So keywords: Felixstowe, Suffolk, UK, seafront, industrial cranes, pier, Ferris wheel, street lamps, waterfront, Europe, England, GBR, port, coastal, daylight, string lights, rocks. Let's make sure they are unique.

Now, let's structure each part:

Title: "Felixstowe Seafront with Industrial Cranes"

Description: "Seafront view of Felixstowe, England, showcasing industrial cranes, a pier, Ferris wheel, and waterfront with street lamps under bright daylight conditions."

Keywords: Felixstowe, Suffolk, UK, seafront, industrial cranes, pier, Ferris wheel, street lamps, waterfront, Europe,
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words. Looking at the image: seafront, Felixstowe, England, with cranes, pier, Ferris wheel. So maybe "Felixstowe Seafront with Industrial Cranes". 

Then the description: need to combine context (date/time, location) with main subjects (seafront, cranes, pier, Ferris wheel, lighting, action). The image has a seafront with industrial cranes, a pier, Ferris wheel, street lamps, string lights, water, and a person maybe. So something like "Seafront view of Felixstowe, England, showing industrial cranes, pier, Ferris wheel, and waterfront with street lamps under bright daylight." Wait, need to include the context like capture date, but maybe not in the description, but the description should mention the setting. Let's make it: "Seafront scene in Felixstowe, England, featuring industrial cranes, a pier, Ferris wheel, and waterfront with street lamps under bright daylight conditions." 

Keywords need 10-18, unique, comma-separated. From hints: East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront, Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. Wait, need unique ones. Let's list: Felixstowe, Suffolk, UK, seafront, industrial cranes, pier, Ferris wheel, street lamps, waterfront, Europe, England. Wait, let's check the image: street lamps, string lights, water, cranes, pier, Ferris wheel, trees, rocks. So keywords: Felixstowe, Suffolk, UK, seafront, industrial cranes, pier, Ferris wheel, street lamps, waterfront, Europe, England, GBR, port, coastal, daylight, string lights, rocks. Let's make sure they are unique. 

Now, let's structure each part:

Title: "Felixstowe Seafront with Industrial Cranes"

Description: "Seafront view of Felixstowe, England, showcasing industrial cranes, a pier, Ferris wheel, and waterfront with street lamps under bright daylight conditions."

Keywords: Felixstowe, Suffolk, UK, seafront, industrial cranes, pier, Ferris wheel, street lamps, waterfront, Europe,
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
  the token limit; Title has 2 words (requested 5-10); Keyword list has 146
  terms (requested 10-18); Duplicate keywords: pier, industrial cranes, street
  lamps, waterfront, bright daylight, pier view, bright daylight scene, street
  lamp view, waterfront trees, industrial cranes view, waterfront view, trees
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.42s
- *Generation time:* 12.90s
- *Total time:* 14.33s
- *Input validation time:* 4.52e-05
- *Prompt preparation time:* 0.00764
- *First-token latency:* 5.53
- *Cleanup time:* 0.121
- *Prompt tokens:* 6,310
- *Generation tokens:* 500
- *Total tokens:* 6,810
- *Prompt throughput (raw):* 1,141 tok/s
- *Generation throughput (raw):* 72.2 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.42
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00211
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1213
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
Title: Felixstowe Seafront
Description: A view of the Felixstowe seafront, featuring a pier, industrial cranes, street lamps, and a tree-lined waterfront under bright daylight.
Keywords: Felixstowe seafront, England UK, pier, industrial cranes, street lamps, waterfront, bright daylight, East Suffolk, Suffolk, UK, gbr, Europe, seafront view, pier view, industrial waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial
</pre>

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
  the Title field; Response appears cut off at the token limit; Internal
  reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 3.25s
- *Generation time:* 112.15s
- *Total time:* 115.41s
- *Input validation time:* 4.86e-05
- *Prompt preparation time:* 0.00177
- *First-token latency:* 1.61
- *Cleanup time:* 0.138
- *Prompt tokens:* 1,257
- *Generation tokens:* 500
- *Total tokens:* 1,757
- *Prompt throughput (raw):* 781 tok/s
- *Generation throughput (raw):* 4.55 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.5
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00365
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1319
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

First, the context says the capture date/time is 2026-08-13 16:08:57 UTC+01:00, and GPS is 51.961800°N, 1.358500°E. The location is Felixstowe, England, UK, GBR, Europe. The hints mention seafront, Felixstowe, England, UK, GBR, seafront, etc.

Looking at the image: there's a seafront with a pier, lampposts, a Ferris wheel, industrial cranes in the background, water, trees, and a person paddleboarding. The lighting suggests it's daytime, maybe sunny. The scene is Felixstowe's seafront.

Now, the title needs to be a concrete 5-10 words. Let's think: "Felixstowe seafront with pier, lampposts, and paddleboarder at 16:08 UTC+01:00". Wait, but the time is in the future (2026), so maybe just the location and key elements. Maybe "Felixstowe seafront at 16:08 UTC+01:00 with pier, lampposts, and paddleboarder".

Description: Combine context and visible details. So, "A daytime view of Felixstowe's seafront at 16:08 UTC+01:00, featuring a pier, traditional lampposts, a paddleboarder on the water, industrial cranes in the distance, and trees framing the scene."

Keywords: Need 10-18 unique keywords. Let's list: Felixstowe, seafront, England, UK, GBR, UTC+01:00, 2026-08-13, pier, lampposts, paddleboarder, industrial cranes, Ferris wheel, trees, water, Europe.

Wait, check the hints: the hints include Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. But the image might not show Adobe Stock or Any Vision. The context is authoritative, so maybe include those. Wait, the user said to treat capture date/time and GPS as authoritative, but don't claim they are visible. So maybe include them as
</pre>

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
- *Model load time:* 2.24s
- *Generation time:* 25.21s
- *Total time:* 27.46s
- *Input validation time:* 4.11e-05
- *Prompt preparation time:* 0.00472
- *First-token latency:* 2.53
- *Cleanup time:* 0.108
- *Prompt tokens:* 3,100
- *Generation tokens:* 131
- *Total tokens:* 3,231
- *Prompt throughput (raw):* 1,225 tok/s
- *Generation throughput (raw):* 5.98 tok/s
- *Peak memory:* 24
- *Active memory:* 19
- *Cache memory:* 5.0
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 0.00655
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1170
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
Felixstowe Seafront: Ferris Wheel, Pier, and Industrial Port

A scenic Felixstowe seafront view captures a Ferris wheel, pier, and industrial port under a clear sky. The image shows a pier extending into the water, a Ferris wheel in the distance, and a port with cranes and buildings, creating a contrast between leisure and industry.

Felixstowe, England, UK, GBR, seafront, Ferris wheel, pier, industrial port, cranes, buildings, sky, water, pier lights, trees, rocks, summer, daylight, coastal, East Suffolk, Any Vision, Adobe Stock
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
- *Model load time:* 0.73s
- *Generation time:* 24.29s
- *Total time:* 25.04s
- *Input validation time:* 5.23e-05
- *Prompt preparation time:* 0.0117
- *First-token latency:* 18
- *Cleanup time:* 0.128
- *Prompt tokens:* 16,469
- *Generation tokens:* 500
- *Total tokens:* 16,969
- *Prompt throughput (raw):* 915 tok/s
- *Generation throughput (raw):* 90.2 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.4
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00806
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1212
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
Got it, let's tackle this step by step. First, the title. The context says Felixstowe, England, UK, GBR. The image has a seafront with cranes, a Ferris wheel, street lamps. So a title should be specific. Maybe "Felixstowe Seafront, England, UK" but need 5-10 words. Let's see: "Felixstowe Seafront, England, UK, Cranes and Ferris Wheel" – that's 10 words. Wait, "Felixstowe Seafront, England, UK" is 7, but maybe "Felixstowe Seafront, England, UK: Cranes and Ferris Wheel" – no, better to be concise. "Felixstowe Seafront, England, UK, Cranes and Ferris Wheel" – 10 words. That works.

Next, the description. Needs to combine context with visible subject, setting, action, lighting, details. Context: capture date/time, GPS, but we don't claim those. The main visible subject is the seafront with cranes, Ferris wheel, street lamps. Setting: coastal, industrial port. Action: maybe a person on paddleboard? Wait, the image has a person on a board in the water. Lighting: sunny, clear sky. Distinctive details: colorful string lights, old-fashioned street lamps, Ferris wheel, cranes. So: "Sunny seafront at Felixstowe, England, UK, featuring cranes, Ferris wheel, colorful string lights, and vintage street lamps over calm waters with a paddleboarder."

Wait, check the context: the description hint says "Seafront, Felixstowe, England, UK, GBR". So need to include that. Also, the GPS is 51.9618N, 1.3585E, which is Felixstowe, Suffolk, UK. So the description should mention Felixstowe, England, UK, GBR. Let's structure: "Felixstowe seafront, England, UK, featuring cranes, Ferris wheel, colorful string lights, vintage street lamps, and a paddleboarder on calm waters under a clear sky."

Now keywords. Need 10-18 unique, comma-separated. Context: Felixstowe, England, UK, GBR, Suffolk. Visible details: seafront, cr
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
  the token limit; Keyword list has 122 terms (requested 10-18); Duplicate
  keywords: uk, europe, gbr, uk seafront, uk waterfront, uk ferris wheel, uk
  cranes, uk sunny day, uk east suffolk, uk felixstowe, uk east anglia, uk
  industrial port
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.21s
- *Generation time:* 28.63s
- *Total time:* 29.84s
- *Input validation time:* 5.43e-05
- *Prompt preparation time:* 0.000296
- *First-token latency:* 18.6
- *Cleanup time:* 0.0931
- *Prompt tokens:* 16,478
- *Generation tokens:* 500
- *Total tokens:* 16,978
- *Prompt throughput (raw):* 887 tok/s
- *Generation throughput (raw):* 53.7 tok/s
- *Peak memory:* 13
- *Active memory:* 9.5
- *Cache memory:* 1.3
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1262
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
Title: Felixstowe Seafront, East Suffolk, UK

Description: A sunny view of Felixstowe's seafront, featuring a pier, Ferris wheel, cranes, and decorative string lights, with a lamp post in the foreground and a person paddleboarding in the distance.

Keywords: Felixstowe, East Suffolk, England, UK, Europe, seafront, pier, Ferris wheel, cranes, string lights, lamp post, paddleboarding, sunny, waterfront, harbor, industrial, coastal, summer, vacation, travel, landscape, scenic, scenic view, waterfront activity, East Anglia, GBR, United Kingdom, GBR, Great Britain, Europe, British, English, UK, GBR, sea, water, blue, sky, trees, rocks, rocks by the sea, waterfront promenade, East Suffolk coastline, East Anglia seafront, UK seafront, UK waterfront, UK coastal scene, UK industrial landscape, UK scenic view, UK travel destination, UK summer scene, UK waterfront activity, UK Ferris wheel, UK cranes, UK lamp post, UK string lights, UK sunny day, UK paddleboarding, UK waterfront, UK seafront, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK Ferris wheel, UK cranes, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day,
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
- *Observations:* Missing or empty fields: Title, Description; Extra text
  appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Model load time:* 2.36s
- *Generation time:* 8.80s
- *Total time:* 11.16s
- *Input validation time:* 4.51e-05
- *Prompt preparation time:* 0.00321
- *First-token latency:* 0.378
- *Cleanup time:* 0.121
- *Prompt tokens:* 568
- *Generation tokens:* 391
- *Total tokens:* 959
- *Prompt throughput (raw):* 1,501 tok/s
- *Generation throughput (raw):* 48.7 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0114
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1189
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
## Image Description

The image captures a scenic view of a waterfront area, likely a harbor or bay, on a bright day. The foreground is dominated by several black lampposts, with the closest one sharply in focus. The lampposts are ornate, featuring a classic lantern design.

Behind the lampposts, a body of water stretches out, reflecting the bright sky. A long pier extends across the water, adorned with strings of lights, suggesting a festive or decorative purpose. In the distance, a large Ferris wheel stands prominently, adding a touch of amusement park charm to the scene.

Further in the background, industrial structures, including large cranes, are visible along the shoreline. These structures indicate a working port or industrial area. The overall atmosphere is one of a bustling waterfront with a mix of recreational and industrial activity. The sky is a pale blue, suggesting clear weather.

## Contextual Information

**Date/Time:** August 13, 2026, 16:08:57 UTC+01:00
**Location:** 51.961800°N, 1.358500°E (This coordinates point to Felixstowe, England, UK)

**Keywords:** waterfront, harbor, pier, Ferris wheel, cranes, industrial, lights, lampposts, sea, sky, England, UK, Felixstowe, coastal, urban, summer, daytime.

## Additional Details

The image appears to be taken from a slightly elevated position, offering a panoramic view of the waterfront. The focus is primarily on the foreground lampposts, with the background gradually softening into a slightly hazy view. The lighting suggests it is daytime, with ample sunlight illuminating the scene. The presence of the Ferris wheel and decorative lights on the pier hint at a location that caters to both leisure and industrial activities.
</pre>

<details>
<summary>Exact raw output</summary>

```text
## Image Description

The image captures a scenic view of a waterfront area, likely a harbor or bay, on a bright day. The foreground is dominated by several black lampposts, with the closest one sharply in focus. The lampposts are ornate, featuring a classic lantern design. 

Behind the lampposts, a body of water stretches out, reflecting the bright sky. A long pier extends across the water, adorned with strings of lights, suggesting a festive or decorative purpose. In the distance, a large Ferris wheel stands prominently, adding a touch of amusement park charm to the scene. 

Further in the background, industrial structures, including large cranes, are visible along the shoreline. These structures indicate a working port or industrial area. The overall atmosphere is one of a bustling waterfront with a mix of recreational and industrial activity. The sky is a pale blue, suggesting clear weather. 

## Contextual Information

**Date/Time:** August 13, 2026, 16:08:57 UTC+01:00
**Location:** 51.961800°N, 1.358500°E (This coordinates point to Felixstowe, England, UK)

**Keywords:** waterfront, harbor, pier, Ferris wheel, cranes, industrial, lights, lampposts, sea, sky, England, UK, Felixstowe, coastal, urban, summer, daytime. 

## Additional Details

The image appears to be taken from a slightly elevated position, offering a panoramic view of the waterfront. The focus is primarily on the foreground lampposts, with the background gradually softening into a slightly hazy view. The lighting suggests it is daytime, with ample sunlight illuminating the scene. The presence of the Ferris wheel and decorative lights on the pier hint at a location that caters to both leisure and industrial activities.
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
- *Model load time:* 0.96s
- *Generation time:* 4.62s
- *Total time:* 5.58s
- *Input validation time:* 4.79e-05
- *Prompt preparation time:* 0.00155
- *First-token latency:* 2.4
- *Cleanup time:* 0.0856
- *Prompt tokens:* 2,680
- *Generation tokens:* 102
- *Total tokens:* 2,782
- *Prompt throughput (raw):* 1,117 tok/s
- *Generation throughput (raw):* 62.2 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.90
- *Model-load active memory:* 8.05
- *Post-cleanup active memory:* 0.0125
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1134
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
Seafront, Felixstowe, England, UK, GBR, Europe, 2026-08-13 16:08:57 UTC+01:00, 51.961800°N, 1.358500°E, Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront
</pre>

<details>
<summary>Exact raw output</summary>

```text
Seafront, Felixstowe, England, UK, GBR, Europe, 2026-08-13 16:08:57 UTC+01:00, 51.961800°N, 1.358500°E, Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront 
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
- *Model load time:* 0.50s
- *Generation time:* 1.00s
- *Total time:* 1.50s
- *Input validation time:* 4.9e-05
- *Prompt preparation time:* 0.00135
- *First-token latency:* 0.0798
- *Cleanup time:* 0.0795
- *Prompt tokens:* 304
- *Generation tokens:* 171
- *Total tokens:* 475
- *Prompt throughput (raw):* 3,812 tok/s
- *Generation throughput (raw):* 353 tok/s
- *Peak memory:* 2.2
- *Active memory:* 0.61
- *Cache memory:* 0.30
- *Model-load active memory:* 0.611
- *Post-cleanup active memory:* 0.0131
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1218
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
Title: Seafront, Felixstowe, England, UK, GBR, Europe
Description: A 5-10-word title that describes the setting and context of the image, with a focus on the maritime theme. It also includes a 1-2-sentence description that combines relevant information about the location, such as the name of the town, the country, and the GBR (Great Britain) code, which is a standard for maritime identification in the UK. The description also includes the main visible subject, which is the lamp post, and the distinctive details, such as the lighting and the silhouette of the lamp post. The keywords, which cover relevant context and visible details, are: "Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront."
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
- *Model load time:* 1.20s
- *Generation time:* 14.05s
- *Total time:* 15.26s
- *Input validation time:* 3.96e-05
- *Prompt preparation time:* 0.000165
- *First-token latency:* 1.1
- *Cleanup time:* 0.187
- *Prompt tokens:* 4,397
- *Generation tokens:* 500
- *Total tokens:* 4,897
- *Prompt throughput (raw):* 4,007 tok/s
- *Generation throughput (raw):* 40.8 tok/s
- *Peak memory:* 4.4
- *Active memory:* 1.7
- *Cache memory:* 1.9
- *Model-load active memory:* 1.73
- *Post-cleanup active memory:* 0.0141
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1118
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
- Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, GBR,
- Description hint: Seafront, Felixstowe, England, UK, GBR,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
-
</pre>

</details>

---
