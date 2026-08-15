# Model Output Gallery

Generated on: 2026-08-16 00:48:24 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|------------|-----------------|---------|------------|---------|--------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | 2.21s   | 503 tok/s  | 0.05            | 1.1     | 400        | 98      | none                                                                                                               |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | 11.16s  | 15.0 tok/s | 1.72            | 15      | 314        | 106     | none                                                                                                               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 7.70s   | 58.3 tok/s | 3.22            | 14      | 3,228      | 143     | none                                                                                                               |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 3.60s   | 172 tok/s  | 1.36            | 9.0     | 3,227      | 104     | none                                                                                                               |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | 113.30s | 48.9 tok/s | 93.94           | 74      | 16,522     | 105     | none                                                                                                               |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | 4.44s   | 50.9 tok/s | 0.32            | 9.4     | 1,137      | 110     | none                                                                                                               |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 59.31s  | 90.4 tok/s | 51.74           | 24      | 16,522     | 116     | none                                                                                                               |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | 125.25s | 13.0 tok/s | 111.68          | 35      | 16,522     | 114     | none                                                                                                               |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 5.15s   | 91.3 tok/s | 0.45            | 16      | 619        | 98      | none                                                                                                               |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 10.04s  | 16.3 tok/s | 1.46            | 20      | 619        | 73      | none                                                                                                               |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | 14.63s  | 22.2 tok/s | 5.50            | 24      | 2,695      | 95      | title/keyword constraints failed                                                                                   |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | 14.03s  | 67.3 tok/s | 10.73           | 8.4     | 6,348      | 81      | title/keyword constraints failed                                                                                   |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | 50.48s  | 26.7 tok/s | 36.04           | 78      | 6,348      | 83      | control tokens visible; title/keyword constraints failed                                                           |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | 10.10s  | 27.6 tok/s | 1.66            | 18      | 2,626      | 141     | role tokens visible; title/keyword constraints failed                                                              |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable_with_caveats` | 7.97s   | 25.9 tok/s | 1.96            | 17      | 3,662      | 86      | title/keyword constraints failed                                                                                   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | 7.81s   | 60.6 tok/s | 3.28            | 15      | 3,228      | 145     | title/keyword constraints failed                                                                                   |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | 5.17s   | 48.3 tok/s | 0.69            | 11      | 1,534      | 106     | title/keyword constraints failed                                                                                   |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable_with_caveats` | 107.81s | 186 tok/s  | 105.63          | 5.1     | 16,517     | 102     | title/keyword constraints failed; draft hints copied unchanged                                                     |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable_with_caveats` | 67.65s  | 83.8 tok/s | 63.61           | 10.0    | 16,522     | 96      | title/keyword constraints failed                                                                                   |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable_with_caveats` | 39.54s  | 36.1 tok/s | 24.56           | 70      | 3,503      | 106     | title/keyword constraints failed                                                                                   |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | 6.05s   | 62.0 tok/s | 0.35            | 29      | 615        | 85      | control tokens visible                                                                                             |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | 5.61s   | 66.2 tok/s | 0.34            | 28      | 615        | 82      | control tokens visible                                                                                             |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | 9.05s   | 25.2 tok/s | 1.13            | 18      | 608        | 125     | title/keyword constraints failed                                                                                   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | 9.44s   | 34.1 tok/s | 3.59            | 16      | 3,466      | 107     | title/keyword constraints failed                                                                                   |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `unusable`            | 20.17s  | 86.8 tok/s | 16.34           | 8.4     | 16,506     | 118     | repeated text; title/keyword constraints failed                                                                    |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 36.13s  | 35.2 tok/s | 3.09            | 15      | 3,557      | 1,000   | missing required fields; echoes instructions; cut off at token limit                                               |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 57.57s  | 29.1 tok/s | 6.13            | 60      | 1,627      | 1,000   | extra text before Title; title/keyword constraints failed                                                          |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 3.61s   | 290 tok/s  | 0.11            | 2.1     | 347        | 78      | missing required fields                                                                                            |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 42.63s  | 34.0 tok/s | 10.71           | 13      | 6,344      | 1,000   | repeated text; missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | 189.23s | 4.56 tok/s | 1.77            | 40      | 1,296      | 834     | repeated text; missing required fields; extra text before Title; role tokens visible                               |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `unusable`            | 7.89s   | 164 tok/s  | 0.11            | 4.1     | 603        | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                            |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | 2.21s   | 247 tok/s  | 0.28            | 3.7     | 1,004      | 41      | missing required fields; extra text before Title                                                                   |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | 24.23s  | 6.27 tok/s | 2.89            | 24      | 3,158      | 113     | missing required fields                                                                                            |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `unusable`            | 32.66s  | 92.1 tok/s | 29.51           | 8.4     | 16,506     | 118     | repeated text; title/keyword constraints failed                                                                    |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | 38.17s  | 80.9 tok/s | 25.23           | 8.4     | 16,508     | 898     | extra text before Title; title/keyword constraints failed                                                          |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | 49.21s  | 45.8 tok/s | 25.14           | 13      | 16,517     | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                            |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 9.93s   | 40.2 tok/s | 0.38            | 17      | 607        | 261     | missing required fields                                                                                            |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | 6.14s   | 57.3 tok/s | 3.27            | 9.7     | 2,734      | 53      | missing required fields                                                                                            |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | 2.54s   | 186 tok/s  | 0.09            | 2.1     | 343        | 81      | missing required fields                                                                                            |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | 26.82s  | 32.1 tok/s | 1.63            | 4.3     | 4,435      | 726     | repeated text; missing required fields; echoes instructions                                                        |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `not_evaluated`       | 2.94s   | -          | -               | -       | -          | -       | none                                                                                                               |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 503 tok/s

Average clean-completion throughput: 106 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.1 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Usability       | Observations                                                                                                       |
|--------------------------------------------------------------------------------------------------------------|-----------------|--------------------------------------------------------------------------------------------------------------------|
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                              | `unusable`      | repeated text; title/keyword constraints failed                                                                    |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable`      | missing required fields; echoes instructions; cut off at token limit                                               |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable`      | extra text before Title; title/keyword constraints failed                                                          |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                   | `unusable`      | repeated text; missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | `unusable`      | repeated text; missing required fields; extra text before Title; role tokens visible                               |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                            |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                 | `unusable`      | missing required fields; extra text before Title                                                                   |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                  | `unusable`      | repeated text; title/keyword constraints failed                                                                    |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                  | `unusable`      | extra text before Title; title/keyword constraints failed                                                          |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                            |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                            | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                 | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable`      | repeated text; missing required fields; echoes instructions                                                        |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                 | `not_evaluated` | none                                                                                                               |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | Title:<br>Dover Castle, Dover, England, UK, GBR, Europe<br><br>Description:<br>An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.<br><br>Keywords:<br>Dover Castle, England,... |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | Title: Historic Dover Castle Perched on a Steep Hillside<br><br>Description: The medieval stone castle, situated in Dover, England, boasts a striking architecture featuring round towers and an arched entranceway, set against a partly cloudy sky. Built on a steep hillside, the castle...                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>**Dover Castle’s Medieval Stone Keep and Arched Gateway**<br><br>Description:<br>A dramatic exterior view of Dover Castle’s fortified stone keep, showcasing its robust round towers and grand arched entranceway, perched on a steep, verdant hillside under soft, partly cloudy Augus...          |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>Dover Castle’s Round Tower Gateway, Kent<br><br>Description:<br>A historic medieval stone fortress captured on a steep grassy hill under partly cloudy skies, featuring robust round towers, an arched entrance, and a small stone bridge—illuminated by natural light in late afternoo...          |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | Title: Dover Castle Gatehouse and Round Towers<br>Description: The historic Dover Castle in Kent, England, features massive medieval stone round towers and a brick gatehouse with an arched entrance. A wooden footbridge connects to the fortress, which rises above a steep, grassy e...                   |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | Title: Dover Castle, Historic Medieval Fortress<br><br>Description: A majestic medieval stone castle with round towers and an arched entranceway, perched on a steep grassy hill, showcases the rich history of Dover, England. The castle's architecture, set against a partly cloudy sky,...                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: Dover Castle Gatehouse and Round Towers<br><br>Description: This exterior view captures the historic medieval stone gatehouse of Dover Castle, featuring prominent round towers and an arched entranceway set against a partly cloudy sky. The fortress stands atop a steep, grassy...                 |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | Title:<br>Exterior view of Dover Castle in Kent, England<br><br>Description:<br>A low-angle exterior view of Dover Castle in Kent, England, showcasing its massive medieval stone walls, round towers, and a prominent arched gateway. The fortress sits atop a steep, grassy hill under a partly...          |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: Historic stone towers of Dover Castle in Kent<br>Description: An exterior view of the medieval stone towers and arched entranceway of Dover Castle, built on a steep grassy hill under a partly cloudy sky. The ancient architecture features massive stone walls and round tower...                   |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: Dover Castle in Dover, Kent, England<br>Description: An exterior view of the historic medieval stone fortress featuring round towers and an arched entranceway situated on a steep grassy hill.<br>Keywords: Dover Castle, England, Kent, UK, medieval, fortress, stone, tower, arch...                |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | Title:<br>Dover Castle, England<br><br>Description:<br>A historic medieval stone castle with round towers and an arched entrance, perched on a steep grassy hill under a partly cloudy sky.<br><br>Keywords:<br>Dover Castle, England, UK, medieval castle, stone architecture, round towers, arched entra... |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | Title: Dover Castle Exterior<br>Description: An exterior view of historic medieval stone Dover Castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.<br>Keywords: Dover Castle, England, Europe, Kent, stone c...                |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: Dover Castle, Kent, England<br>Description: An exterior view of Dover Castle, a historic medieval stone fortress with round towers and an arched entrance, perched on a steep grassy hill under a partly cloudy sky.<br>Keywords: Dover Castle, England, UK, Kent, C...              |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | Title: Dover Castle, England, UK, Medieval Stone Castle on a Hill.<br>Description: Dover Castle, a historic medieval stone castle, features round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky. The castle's architecture i...                   |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable_with_caveats` | Title: Dover Castle, Historic England<br>Description: A medieval stone castle with round towers and an arched entranceway stands on a grassy hill under a partly cloudy sky, captured in 2026.<br>Keywords: Dover Castle, England, Castle, Medieval, Stone, Towers, Arched Entrance, Hill,...                 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | **Title:**<br>*Dover Castle’s Medieval Towers at Sunset*<br><br>**Description:**<br>A historic stone fortress in Dover, England, captured on **August 15, 2026 (UTC+01:00)**, showcases its robust round towers, arched gateway, and steep hillside vegetation under a partly cloudy sky. Built w...          |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | Title: Dover Castle, Dover, England, UK, GBR, Europe<br><br>Description: A medieval stone fortress perched on a steep grassy hill, featuring round towers, an arched entranceway, and a small bridge, built on a partly cloudy day with ancient architecture and blue skies visible.<br><br>Keywo...          |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable_with_caveats` | Title: Dover Castle, Dover, England, UK, GBR, Europe<br>Description: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.<br>Keywords: Adobe Stock, Any Vision,...                |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable_with_caveats` | Title: Dover Castle, Kent, England, UK<br><br>Description: An exterior view of Dover Castle, a historic medieval fortress on a grassy hill, featuring stone towers, an arched gateway, and a small bridge under a partly cloudy sky.<br><br>Keywords: Adobe Stock, Any Vision, Arch, Britain, Cas...          |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable_with_caveats` | Title:<br>Dover Castle medieval stone fortress on grassy hill<br><br>Description:<br>An exterior view of historic Dover Castle in Kent, England, featuring round stone towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.<br><br>Keywords:<br>Dove... |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Exterior view of Dover Castle in Kent, England<br>Description: An exterior view of the medieval stone fortifications of Dover Castle, featuring round towers and an arched entranceway atop a steep grassy hill under a partly cloudy sky.<br>Keywords:...            |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Medieval Stone Towers of Dover Castle in Kent England<br>Description: An exterior view of the historic medieval stone castle featuring round towers and an arched entranceway situated on a steep grassy hill under a cloudy sky.<br>Keywords: Dover Cas...           |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | Title: Dover Castle, Kent: Exterior View, August 2026<br>Description: Dover Castle in Kent is pictured from an exterior angle on 15 August 2026, showing its stone walls and towers rising from a steep, grassy hillside under a partly cloudy sky. A prominent arched entranceway leads...                   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | Title:<br>Dover Castle, England, United Kingdom<br><br>Description:<br>Dover Castle, located in Dover, England, stands majestically on a grassy hill, showcasing its medieval stone architecture and arched entrance under a partly cloudy sky.<br><br>Keywords:<br>Adobe Stock, Arch, Britain, Castle, Do... |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `unusable`            | Title:<br>Dover Castle Exterior<br><br>Description:<br>An exterior view of Dover Castle, a historic medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a small bridge under a partly cloudy sky.<br><br>Keywords: Dover Castle, England, UK, Europe, Fortress...    |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>The user wants British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS. The descriptive hints are given. The user wants:<br><br>- a concrete 5-10-word title;<br>- a 1-2-sentence factual descript...          |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | Alright, let's get to work. I've got the image of Dover Castle in my sights, and I need to generate some metadata. First things first, I'll break down the request. It's a straightforward task: title, description, and keywords. The image is of Dover Castle, and I have the conte...                      |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | Dover Castle, England, UK, GBR, Europe, Medieval, Round Towers, Arched Entranceway, Small Bridge, steep hill, partly cloudy sky, Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, an...                      |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The image is of Dover Castle, so maybe "Dover Castle Exterior View". Then the description should combine context (date/time, location) with the main subject, setting, etc. Th...                |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | ◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords.<br><br>Starting with the title: The image shows Dover Castle, w...                |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `unusable`            | Title: Dover Castle, England, UK, Europe, Fortress, Medieval, Stone, Tower, Hill, Sky, Architecture, Blue<br><br>Description: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a...                |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | Ancient stone castle on hillside under clear sky.<br><br>Keywords: castle, stone, hill, medieval, arch, tower, bridge, UK, Europe, grassy, sky, fortress, architecture                                                                                                                                        |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | Dover Castle: Medieval Stone Fortress on Steep Hillside<br><br>A historic stone castle with round towers and arched entranceway stands atop a steep, grassy hill in Dover, England. Built of brown and white stones, the fortress features a small bridge and is set against a partly cloud...                |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `unusable`            | Title:<br>Dover Castle Exterior<br><br>Description:<br>An exterior view of Dover Castle, a historic medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a small bridge under a partly cloudy sky.<br><br>Keywords: Dover Castle, England, UK, Europe, Fortress...    |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | Got it, let's tackle this step by step. First, the title. The context says Dover Castle, so the title should be specific. The hint says Dover Castle, so "Dover Castle, England" is good. Need 5-10 words. Let's see: "Dover Castle, England, Historic Medieval Stone Castle on Hill"...                      |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | Title:<br>Dover Castle, Kent, England<br><br>Description:<br>A historic medieval stone castle with round towers, an arched entranceway, and a small bridge, situated on a steep grassy hill under a partly cloudy sky, captured on 2026-08-15 at 15:59:46 UTC+01:00.<br><br>Keywords: Dover Castle, Ken...    |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | ```json<br>{<br>"context": {<br>"title": "Dover Castle",<br>"location": "Dover, England, UK, GBR, Europe",<br>"description": "An exterior view of a historic medieval stone castle, featuring round towers, an arched entrance, and a steep grassy hill under a partly cloudy sky."<br>},<br>"date_time":...  |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | Dover Castle, Dover, England, UK, GBR, Europe, historic, medieval, stone, castle, round towers, arched entranceway, steep grassy hill, partly cloudy sky, fortress, ancient, architecture, blue                                                                                                               |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | Title: Dover Castle, England, UK, GBR, Europe<br>Description: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky. The castle is situated on a hill under a...                    |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | - Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue<br>- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, Un...                   |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `not_evaluated`       | Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: Image features and image tokens do not match: tokens: 81, features 1053                                                                                                                                                                 |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.14`
- `mlx`: `0.32.1.dev20260815+9ab977b56`
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

- *Title:* , Dover Castle, Dover, England, UK, GBR, Europe
- *Description:* An exterior view of a historic medieval stone castle,
  featuring round towers, an arched entranceway, and a small bridge, built on
  a steep grassy hill under a partly cloudy sky.
- *Keywords:* Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle,
  England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United
  Kingdom, Wall, ancient, architecture, blue, bridge, british, building,
  defensive, dover, english, exterior, fort, fortification, grass, great,
  green, heritage, historic, history, keep, kingdom, landmark, landscape,
  medieval, moat, nature, old, outdoor, outdoors, ruin, scenic, stronghold,
  summer, tourism, travel, united, view, windows
- *Date:* 2026-08-15 15:59:46 UTC+01:00
- *Time:* 15:59:46
- *GPS:* 51.128800°N, 1.319100°E

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
> &#45; Capture date/time: 2026-08-15 15:59:46 UTC+01:00
> &#45; GPS: 51.128800°N, 1.319100°E
>
> &#8203;Descriptive hints:
> &#45; Title hint: Dover Castle, Dover, England, UK, GBR, Europe
> &#45; Description hint: An exterior view of a historic medieval stone castle,
> featuring round towers, an arched entranceway, and a small bridge, built on
> a steep grassy hill under a partly cloudy sky.
> &#45; Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover
> Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United
> Kingdom, Wall, ancient, architecture, blue
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
- *Model load time:* 0.88s
- *Generation time:* 0.83s
- *Total time:* 2.21s
- *Input validation time:* 0.501
- *Prompt preparation time:* 0.00715
- *First-token latency:* 0.0534
- *Cleanup time:* 0.0843
- *Prompt tokens:* 400
- *Generation tokens:* 98
- *Total tokens:* 498
- *Prompt throughput (raw):* 7,486 tok/s
- *Generation throughput (raw):* 503 tok/s
- *Peak memory:* 1.1
- *Active memory:* 0.90
- *Cache memory:* 0.01
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1399
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
Dover Castle, Dover, England, UK, GBR, Europe

Description:
An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.

Keywords:
Dover Castle, England, UK, GBR, Europe, castle, medieval, stone, tower, arched entranceway, hill, grassy hill, sky, blue
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
- *Model load time:* 1.67s
- *Generation time:* 9.48s
- *Total time:* 11.16s
- *Input validation time:* 0.000165
- *Prompt preparation time:* 0.00162
- *First-token latency:* 1.72
- *Cleanup time:* 0.127
- *Prompt tokens:* 314
- *Generation tokens:* 106
- *Total tokens:* 420
- *Prompt throughput (raw):* 183 tok/s
- *Generation throughput (raw):* 15.0 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.59
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.00431
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1452
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
Title: Historic Dover Castle Perched on a Steep Hillside

Description: The medieval stone castle, situated in Dover, England, boasts a striking architecture featuring round towers and an arched entranceway, set against a partly cloudy sky. Built on a steep hillside, the castle's imposing structure commands a sense of grandeur and history.

Keywords: Dover Castle, medieval, stone, castle, round towers, arched entranceway, steep hillside, partly cloudy sky, historic, architecture, grandeur, history.
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
- *Model load time:* 1.35s
- *Generation time:* 6.33s
- *Total time:* 7.70s
- *Input validation time:* 8.62e-05
- *Prompt preparation time:* 0.00694
- *First-token latency:* 3.22
- *Cleanup time:* 0.101
- *Prompt tokens:* 3,228
- *Generation tokens:* 143
- *Total tokens:* 3,371
- *Prompt throughput (raw):* 1,004 tok/s
- *Generation throughput (raw):* 58.3 tok/s
- *Peak memory:* 14
- *Active memory:* 8.0
- *Cache memory:* 0.65
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.0051
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3786
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
**Dover Castle’s Medieval Stone Keep and Arched Gateway**

Description:
A dramatic exterior view of Dover Castle’s fortified stone keep, showcasing its robust round towers and grand arched entranceway, perched on a steep, verdant hillside under soft, partly cloudy August lighting. The structure’s medieval architecture, with narrow windows and weathered stonework, reflects its historical significance as a key British fortress.

Keywords:
Dover Castle, medieval fortress, stone architecture, round towers, arched gateway, Kent, England, UK, historical landmark, steep hillside, narrow windows, weathered stone, partly cloudy sky, August lighting, British heritage, defensive wall, ancient structure, European history
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
- *Model load time:* 0.97s
- *Generation time:* 2.62s
- *Total time:* 3.60s
- *Input validation time:* 7.02e-05
- *Prompt preparation time:* 0.0069
- *First-token latency:* 1.36
- *Cleanup time:* 0.0777
- *Prompt tokens:* 3,227
- *Generation tokens:* 104
- *Total tokens:* 3,331
- *Prompt throughput (raw):* 2,364 tok/s
- *Generation throughput (raw):* 172 tok/s
- *Peak memory:* 9.0
- *Active memory:* 2.8
- *Cache memory:* 0.44
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00562
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3785
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
Dover Castle’s Round Tower Gateway, Kent

Description:
A historic medieval stone fortress captured on a steep grassy hill under partly cloudy skies, featuring robust round towers, an arched entrance, and a small stone bridge—illuminated by natural light in late afternoon.

Keywords:
ancient architecture, arched entrance, British stone fortress, Dover Castle, England, European medieval wall, fortified hilltop, Kent, round towers, stone bridge, UK, UK castle, medieval stone castle, blue sky lighting
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
- *Model load time:* 12.34s
- *Generation time:* 100.93s
- *Total time:* 113.30s
- *Input validation time:* 8.8e-05
- *Prompt preparation time:* 0.0164
- *First-token latency:* 93.9
- *Cleanup time:* 0.353
- *Prompt tokens:* 16,522
- *Generation tokens:* 105
- *Total tokens:* 16,627
- *Prompt throughput (raw):* 176 tok/s
- *Generation throughput (raw):* 48.9 tok/s
- *Peak memory:* 74
- *Active memory:* 70
- *Cache memory:* 0.88
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.00706
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1439
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
Title: Dover Castle Gatehouse and Round Towers
Description: The historic Dover Castle in Kent, England, features massive medieval stone round towers and a brick gatehouse with an arched entrance. A wooden footbridge connects to the fortress, which rises above a steep, grassy embankment under a partly cloudy sky.
Keywords: Dover Castle, Kent, England, UK, Britain, medieval, fortress, stone, tower, arch, gatehouse, bridge, hill, wall, architecture, ancient, sky, Europe
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
- *Model load time:* 1.05s
- *Generation time:* 3.39s
- *Total time:* 4.44s
- *Input validation time:* 0.000186
- *Prompt preparation time:* 0.0019
- *First-token latency:* 0.321
- *Cleanup time:* 0.112
- *Prompt tokens:* 1,137
- *Generation tokens:* 110
- *Total tokens:* 1,247
- *Prompt throughput (raw):* 3,543 tok/s
- *Generation throughput (raw):* 50.9 tok/s
- *Peak memory:* 9.4
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00713
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1369
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
Title: Dover Castle, Historic Medieval Fortress

Description: A majestic medieval stone castle with round towers and an arched entranceway, perched on a steep grassy hill, showcases the rich history of Dover, England. The castle's architecture, set against a partly cloudy sky, highlights its ancient and enduring presence.

Keywords: Dover Castle, Historic, Medieval, Stone, Tower, Arch, Hill, Grassy, Steep, Partly Cloudy
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
- *Model load time:* 5.16s
- *Generation time:* 54.13s
- *Total time:* 59.31s
- *Input validation time:* 0.000101
- *Prompt preparation time:* 0.0142
- *First-token latency:* 51.7
- *Cleanup time:* 0.125
- *Prompt tokens:* 16,522
- *Generation tokens:* 116
- *Total tokens:* 16,638
- *Prompt throughput (raw):* 319 tok/s
- *Generation throughput (raw):* 90.4 tok/s
- *Peak memory:* 24
- *Active memory:* 21
- *Cache memory:* 0.88
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.00857
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1439
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
Title: Dover Castle Gatehouse and Round Towers

Description: This exterior view captures the historic medieval stone gatehouse of Dover Castle, featuring prominent round towers and an arched entranceway set against a partly cloudy sky. The fortress stands atop a steep, grassy hillside covered in lush vegetation, with a wooden bridge visible on the right side.

Keywords: Dover Castle, England, UK, medieval, stone, architecture, gatehouse, round towers, arched entrance, grassy hill, vegetation, wooden bridge, sky, historic, fortress, Kent, Europe, ancient
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
- *Model load time:* 3.34s
- *Generation time:* 121.88s
- *Total time:* 125.25s
- *Input validation time:* 0.000131
- *Prompt preparation time:* 0.0156
- *First-token latency:* 112
- *Cleanup time:* 0.236
- *Prompt tokens:* 16,522
- *Generation tokens:* 114
- *Total tokens:* 16,636
- *Prompt throughput (raw):* 148 tok/s
- *Generation throughput (raw):* 13.0 tok/s
- *Peak memory:* 35
- *Active memory:* 29
- *Cache memory:* 1.8
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.0096
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1439
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
Exterior view of Dover Castle in Kent, England

Description:
A low-angle exterior view of Dover Castle in Kent, England, showcasing its massive medieval stone walls, round towers, and a prominent arched gateway. The fortress sits atop a steep, grassy hill under a partly cloudy sky, with a wooden bridge visible to the right.

Keywords:
Dover Castle, Kent, England, UK, medieval, fortress, stone, architecture, tower, arch, bridge, hill, sky, historic, ancient, wall, Europe, Britain
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
- *Model load time:* 2.77s
- *Generation time:* 2.34s
- *Total time:* 5.15s
- *Input validation time:* 0.000167
- *Prompt preparation time:* 0.0246
- *First-token latency:* 0.448
- *Cleanup time:* 0.208
- *Prompt tokens:* 619
- *Generation tokens:* 98
- *Total tokens:* 717
- *Prompt throughput (raw):* 1,383 tok/s
- *Generation throughput (raw):* 91.3 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.21
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0118
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1402
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
Title: Historic stone towers of Dover Castle in Kent
Description: An exterior view of the medieval stone towers and arched entranceway of Dover Castle, built on a steep grassy hill under a partly cloudy sky. The ancient architecture features massive stone walls and round towers overlooking the landscape in Kent, England.
Keywords: Dover Castle, Kent, England, United Kingdom, Europe, medieval, castle, stone, tower, fortress, architecture, historic, hill, sky, ancient, wall, arch
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
- *Model load time:* 3.31s
- *Generation time:* 6.72s
- *Total time:* 10.04s
- *Input validation time:* 0.000175
- *Prompt preparation time:* 0.000392
- *First-token latency:* 1.46
- *Cleanup time:* 0.219
- *Prompt tokens:* 619
- *Generation tokens:* 73
- *Total tokens:* 692
- *Prompt throughput (raw):* 425 tok/s
- *Generation throughput (raw):* 16.3 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.80
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0123
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1402
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
Title: Dover Castle in Dover, Kent, England
Description: An exterior view of the historic medieval stone fortress featuring round towers and an arched entranceway situated on a steep grassy hill.
Keywords: Dover Castle, England, Kent, UK, medieval, fortress, stone, tower, architecture, hill, castle, Britain, Europe, ancient, wall, arch
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
- *Observations:* Title has 3 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 3.82s
- *Generation time:* 10.79s
- *Total time:* 14.63s
- *Input validation time:* 0.000191
- *Prompt preparation time:* 0.00928
- *First-token latency:* 5.5
- *Cleanup time:* 0.158
- *Prompt tokens:* 2,695
- *Generation tokens:* 95
- *Total tokens:* 2,790
- *Prompt throughput (raw):* 490 tok/s
- *Generation throughput (raw):* 22.2 tok/s
- *Peak memory:* 24
- *Active memory:* 18
- *Cache memory:* 0.52
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.000968
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1348
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
Dover Castle, England

Description:
A historic medieval stone castle with round towers and an arched entrance, perched on a steep grassy hill under a partly cloudy sky.

Keywords:
Dover Castle, England, UK, medieval castle, stone architecture, round towers, arched entrance, grassy hill, partly cloudy sky, historic fortress, Kent, Europe, blue sky, green hillside, ancient structure, defensive walls, battlements, heritage site
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
- *Observations:* Title has 3 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.43s
- *Generation time:* 12.58s
- *Total time:* 14.03s
- *Input validation time:* 0.000161
- *Prompt preparation time:* 0.008
- *First-token latency:* 10.7
- *Cleanup time:* 0.116
- *Prompt tokens:* 6,348
- *Generation tokens:* 81
- *Total tokens:* 6,429
- *Prompt throughput (raw):* 591 tok/s
- *Generation throughput (raw):* 67.3 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.46
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00211
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1429
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
Title: Dover Castle Exterior
Description: An exterior view of historic medieval stone Dover Castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
Keywords: Dover Castle, England, Europe, Kent, stone castle, historic medieval, round towers, arched entranceway, steep grassy hill, partly cloudy sky.
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
- *Model load time:* 10.55s
- *Generation time:* 39.90s
- *Total time:* 50.48s
- *Input validation time:* 0.00011
- *Prompt preparation time:* 0.0112
- *First-token latency:* 36
- *Cleanup time:* 0.56
- *Prompt tokens:* 6,348
- *Generation tokens:* 83
- *Total tokens:* 6,431
- *Prompt throughput (raw):* 176 tok/s
- *Generation throughput (raw):* 26.7 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00243
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1429
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
&lt;|begin_of_box|&gt;Title: Dover Castle, Kent, England
Description: An exterior view of Dover Castle, a historic medieval stone fortress with round towers and an arched entrance, perched on a steep grassy hill under a partly cloudy sky.
Keywords: Dover Castle, England, UK, Kent, Castle, Fortress, Stone, Tower, Arch, Hill, Sky, Architecture, Ancient, Britain, Europe, Wall&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Dover Castle, Kent, England
Description: An exterior view of Dover Castle, a historic medieval stone fortress with round towers and an arched entrance, perched on a steep grassy hill under a partly cloudy sky.
Keywords: Dover Castle, England, UK, Kent, Castle, Fortress, Stone, Tower, Arch, Hill, Sky, Architecture, Ancient, Britain, Europe, Wall<|end_of_box|>
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
- *Observations:* Conversation-role control tokens remain visible; Keyword
  list has 24 terms (requested 10-18); Duplicate keywords: europe, sky
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 2.28s
- *Generation time:* 7.80s
- *Total time:* 10.10s
- *Input validation time:* 0.000186
- *Prompt preparation time:* 0.00281
- *First-token latency:* 1.66
- *Cleanup time:* 0.4
- *Prompt tokens:* 2,626
- *Generation tokens:* 141
- *Total tokens:* 2,767
- *Prompt throughput (raw):* 1,584 tok/s
- *Generation throughput (raw):* 27.6 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.00269
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1385
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
 Title: Dover Castle, England, UK, Medieval Stone Castle on a Hill.
Description: Dover Castle, a historic medieval stone castle, features round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky. The castle's architecture is characterized by its stone walls and towers, and it is a prominent landmark in Dover, England, UK.
Keywords: castle, dover, england, uk, europe, stone, tower, hill, sky, arch, bridge, grass, cloudy, sky, blue, ancient, architecture, fortification, kent, adobe stock, any vision, united kingdom, europe, gbr.&lt;end_of_utterance&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Title: Dover Castle, England, UK, Medieval Stone Castle on a Hill.
Description: Dover Castle, a historic medieval stone castle, features round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky. The castle's architecture is characterized by its stone walls and towers, and it is a prominent landmark in Dover, England, UK.
Keywords: castle, dover, england, uk, europe, stone, tower, hill, sky, arch, bridge, grass, cloudy, sky, blue, ancient, architecture, fortification, kent, adobe stock, any vision, united kingdom, europe, gbr.<end_of_utterance>
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
- *Model load time:* 1.78s
- *Generation time:* 6.15s
- *Total time:* 7.97s
- *Input validation time:* 0.000198
- *Prompt preparation time:* 0.0025
- *First-token latency:* 1.96
- *Cleanup time:* 0.415
- *Prompt tokens:* 3,662
- *Generation tokens:* 86
- *Total tokens:* 3,748
- *Prompt throughput (raw):* 1,868 tok/s
- *Generation throughput (raw):* 25.9 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.28
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.003
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1353
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
Title: Dover Castle, Historic England
Description: A medieval stone castle with round towers and an arched entranceway stands on a grassy hill under a partly cloudy sky, captured in 2026.
Keywords: Dover Castle, England, Castle, Medieval, Stone, Towers, Arched Entrance, Hill, Sky, Partly Cloudy, Kent, Fortress, Architecture, Ancient, Europe, UK, Stone Wall
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
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.47s
- *Generation time:* 6.34s
- *Total time:* 7.81s
- *Input validation time:* 8.36e-05
- *Prompt preparation time:* 0.000377
- *First-token latency:* 3.28
- *Cleanup time:* 0.0887
- *Prompt tokens:* 3,228
- *Generation tokens:* 145
- *Total tokens:* 3,373
- *Prompt throughput (raw):* 984 tok/s
- *Generation throughput (raw):* 60.6 tok/s
- *Peak memory:* 15
- *Active memory:* 8.4
- *Cache memory:* 0.65
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00536
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3786
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
*Dover Castle’s Medieval Towers at Sunset*

**Description:**
A historic stone fortress in Dover, England, captured on **August 15, 2026 (UTC+01:00)**, showcases its robust round towers, arched gateway, and steep hillside vegetation under a partly cloudy sky. Built with medieval architecture, its imposing walls and small bridge highlight its enduring legacy.

**Keywords:**
Dover Castle, medieval architecture, Kent, England, United Kingdom, stone fortress, round towers, arched gateway, steep hillside, partly cloudy sky, historic landmark, ancient walls, defensive bridge, European heritage, sunset lighting, Gothic design, exterior view, Dover, GBR
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type molmo)
- *Model load time:* 1.18s
- *Generation time:* 3.98s
- *Total time:* 5.17s
- *Input validation time:* 5.48e-05
- *Prompt preparation time:* 0.00165
- *First-token latency:* 0.688
- *Cleanup time:* 0.0853
- *Prompt tokens:* 1,534
- *Generation tokens:* 106
- *Total tokens:* 1,640
- *Prompt throughput (raw):* 2,228 tok/s
- *Generation throughput (raw):* 48.3 tok/s
- *Peak memory:* 11
- *Active memory:* 9.0
- *Cache memory:* 0.14
- *Model-load active memory:* 9.04
- *Post-cleanup active memory:* 0.00593
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1344
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
 Title: Dover Castle, Dover, England, UK, GBR, Europe

Description: A medieval stone fortress perched on a steep grassy hill, featuring round towers, an arched entranceway, and a small bridge, built on a partly cloudy day with ancient architecture and blue skies visible.

Keywords: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
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
- *Observations:* Keyword list has 20 terms (requested 10-18); Title,
  Description and Keywords copy all supplied hints unchanged
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.59s
- *Generation time:* 107.21s
- *Total time:* 107.81s
- *Input validation time:* 8.22e-05
- *Prompt preparation time:* 0.00389
- *First-token latency:* 106
- *Cleanup time:* 0.129
- *Prompt tokens:* 16,517
- *Generation tokens:* 102
- *Total tokens:* 16,619
- *Prompt throughput (raw):* 156 tok/s
- *Generation throughput (raw):* 186 tok/s
- *Peak memory:* 5.1
- *Active memory:* 1.3
- *Cache memory:* 0.86
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.00744
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1478
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
Title: Dover Castle, Dover, England, UK, GBR, Europe
Description: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
Keywords: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
</pre>

</details>

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### mlx-community/Qwen3.5-9B-MLX-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-9B-MLX-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 1.44s
- *Generation time:* 66.20s
- *Total time:* 67.65s
- *Input validation time:* 8.62e-05
- *Prompt preparation time:* 0.000578
- *First-token latency:* 63.6
- *Cleanup time:* 0.146
- *Prompt tokens:* 16,522
- *Generation tokens:* 96
- *Total tokens:* 16,618
- *Prompt throughput (raw):* 260 tok/s
- *Generation throughput (raw):* 83.8 tok/s
- *Peak memory:* 10.0
- *Active memory:* 6.6
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00909
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1439
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
Title: Dover Castle, Kent, England, UK

Description: An exterior view of Dover Castle, a historic medieval fortress on a grassy hill, featuring stone towers, an arched gateway, and a small bridge under a partly cloudy sky.

Keywords: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
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
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 11.00s
- *Generation time:* 28.51s
- *Total time:* 39.54s
- *Input validation time:* 0.000211
- *Prompt preparation time:* 0.0193
- *First-token latency:* 24.6
- *Cleanup time:* 0.275
- *Prompt tokens:* 3,503
- *Generation tokens:* 106
- *Total tokens:* 3,609
- *Prompt throughput (raw):* 143 tok/s
- *Generation throughput (raw):* 36.1 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.41
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.00986
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1426
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
Dover Castle medieval stone fortress on grassy hill

Description:
An exterior view of historic Dover Castle in Kent, England, featuring round stone towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.

Keywords:
Dover Castle, medieval castle, stone fortress, round towers, arched entrance, stone bridge, grassy hill, Kent, England, UK, Britain, Europe, historic architecture, ancient structure, blue sky, partly cloudy, Adobe Stock, wall, tower
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
- *Model load time:* 3.43s
- *Generation time:* 2.59s
- *Total time:* 6.05s
- *Input validation time:* 0.000129
- *Prompt preparation time:* 0.0207
- *First-token latency:* 0.347
- *Cleanup time:* 0.142
- *Prompt tokens:* 615
- *Generation tokens:* 85
- *Total tokens:* 700
- *Prompt throughput (raw):* 1,770 tok/s
- *Generation throughput (raw):* 62.0 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1374
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
&lt;channel|&gt;Title: Exterior view of Dover Castle in Kent, England
Description: An exterior view of the medieval stone fortifications of Dover Castle, featuring round towers and an arched entranceway atop a steep grassy hill under a partly cloudy sky.
Keywords: Dover Castle, England, Kent, UK, castle, fortress, stone, tower, architecture, medieval, ancient, arch, hill, history, Britain, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Exterior view of Dover Castle in Kent, England
Description: An exterior view of the medieval stone fortifications of Dover Castle, featuring round towers and an arched entranceway atop a steep grassy hill under a partly cloudy sky.
Keywords: Dover Castle, England, Kent, UK, castle, fortress, stone, tower, architecture, medieval, ancient, arch, hill, history, Britain, Europe
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
- *Generation time:* 2.44s
- *Total time:* 5.61s
- *Input validation time:* 9.32e-05
- *Prompt preparation time:* 0.000369
- *First-token latency:* 0.34
- *Cleanup time:* 0.131
- *Prompt tokens:* 615
- *Generation tokens:* 82
- *Total tokens:* 697
- *Prompt throughput (raw):* 1,809 tok/s
- *Generation throughput (raw):* 66.2 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1374
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
&lt;channel|&gt;Title: Medieval Stone Towers of Dover Castle in Kent England
Description: An exterior view of the historic medieval stone castle featuring round towers and an arched entranceway situated on a steep grassy hill under a cloudy sky.
Keywords: Dover Castle, England, UK, Britain, Castle, Fortress, Hill, Stone, Tower, Wall, ancient, architecture, Arch, Medieval, Kent, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Medieval Stone Towers of Dover Castle in Kent England
Description: An exterior view of the historic medieval stone castle featuring round towers and an arched entranceway situated on a steep grassy hill under a cloudy sky.
Keywords: Dover Castle, England, UK, Britain, Castle, Fortress, Hill, Stone, Tower, Wall, ancient, architecture, Arch, Medieval, Kent, Europe
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
- *Model load time:* 2.29s
- *Generation time:* 6.75s
- *Total time:* 9.05s
- *Input validation time:* 6.89e-05
- *Prompt preparation time:* 0.00275
- *First-token latency:* 1.13
- *Cleanup time:* 0.123
- *Prompt tokens:* 608
- *Generation tokens:* 125
- *Total tokens:* 733
- *Prompt throughput (raw):* 540 tok/s
- *Generation throughput (raw):* 25.2 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.45
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0107
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1403
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
Title: Dover Castle, Kent: Exterior View, August 2026
Description: Dover Castle in Kent is pictured from an exterior angle on 15 August 2026, showing its stone walls and towers rising from a steep, grassy hillside under a partly cloudy sky. A prominent arched entranceway leads through the castle, with a small bridge visible to the right.
Keywords: Dover Castle, Kent, England, UK, United Kingdom, castle, fortress, architecture, stone, tower, arch, hill, exterior, August, 2026, grassy, bridge, medieval, walls
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
- *Observations:* Keyword list has 25 terms (requested 10-18); Duplicate
  keywords: stone
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.97s
- *Generation time:* 7.45s
- *Total time:* 9.44s
- *Input validation time:* 0.000135
- *Prompt preparation time:* 0.00885
- *First-token latency:* 3.59
- *Cleanup time:* 0.0992
- *Prompt tokens:* 3,466
- *Generation tokens:* 107
- *Total tokens:* 3,573
- *Prompt throughput (raw):* 966 tok/s
- *Generation throughput (raw):* 34.1 tok/s
- *Peak memory:* 16
- *Active memory:* 13
- *Cache memory:* 0.64
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0143
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1348
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
Dover Castle, England, United Kingdom

Description:
Dover Castle, located in Dover, England, stands majestically on a grassy hill, showcasing its medieval stone architecture and arched entrance under a partly cloudy sky.

Keywords:
Adobe Stock, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue, cloudy, grassy, historic, medieval, stone, structure
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
- *Observations:* Response repeats the same text; Title has 3 words (requested
  5-10); Keyword list has 33 terms (requested 10-18); Duplicate keywords:
  england, uk, europe, fortress, castle, stone, tower, wall, hill, sky,
  ancient, architecture, united kingdom
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 1.22s
- *Generation time:* 18.93s
- *Total time:* 20.17s
- *Input validation time:* 7.79e-05
- *Prompt preparation time:* 0.0103
- *First-token latency:* 16.3
- *Cleanup time:* 0.162
- *Prompt tokens:* 16,506
- *Generation tokens:* 118
- *Total tokens:* 16,624
- *Prompt throughput (raw):* 1,010 tok/s
- *Generation throughput (raw):* 86.8 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1420
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
Dover Castle Exterior

Description:
An exterior view of Dover Castle, a historic medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a small bridge under a partly cloudy sky.

Keywords: Dover Castle, England, UK, Europe, Fortress, Castle, Stone, Tower, Wall, Hill, Sky, Arch, Ancient, Architecture, Britain, Kent, United Kingdom, Stone, Wall, Castle, England, Europe, Fortress, Hill, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
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
- *Model load time:* 3.70s
- *Generation time:* 32.41s
- *Total time:* 36.13s
- *Input validation time:* 0.000139
- *Prompt preparation time:* 0.0114
- *First-token latency:* 3.09
- *Cleanup time:* 0.172
- *Prompt tokens:* 3,557
- *Generation tokens:* 1,000
- *Total tokens:* 4,557
- *Prompt throughput (raw):* 1,152 tok/s
- *Generation throughput (raw):* 35.2 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.92
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.000706
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1722
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
The user wants British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS. The descriptive hints are given. The user wants:

- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

We must return exactly three sections: Title:, Description:, Keywords: and nothing else.

We need to treat capture date/time and GPS as authoritative facts, but not claim they are visible. So we can incorporate them in description if needed, but we must not claim they are visible. The description should combine relevant context with main visible subject, setting, action, lighting, and distinctive details.

We need to ensure British English spelling (e.g., "favourite" vs "favorite"? Not needed). Use British English.

We need to incorporate the visible details: The image shows a medieval stone castle with round towers, an arched entrance, a small bridge, built on a steep grassy hill under a partly cloudy sky. The castle is Dover Castle, Dover, England, UK. The image shows stone walls, crenellations, battlements, windows, a gatehouse with an arch, a small bridge leading to the gate, a grassy hill with vegetation, partly cloudy sky.

We need to produce a title of 5-10 words. Something like "Dover Castle on a Grassy Hill". That's 6 words. Or "Dover Castle, Round Towers, Hilltop View". That's 7 words. Or "Dover Castle: Hilltop Stone Fortress". That's 5 words. Or "Dover Castle, Hilltop Stone Fortress". That's 5 words. Or "Dover Castle, Hilltop Medieval Fortress". That's 5 words. Or "Dover Castle, Hilltop Medieval Castle". That's 5 words. Or "Dover Castle, Hilltop Stone Castle". That's 5 words. Or "Dover Castle, Hilltop Stone Fortress". That's 5 words. Let's pick "Dover Castle, Hilltop Stone Fortress". That's 5 words? Let's count: Dover (1) Castle, (2) Hilltop (3) Stone (4) Fortress (5). Yes, 5 words. That's within 5-10.

Now description: 1-2 sentences. Must combine relevant context with main visible subject, setting, action, lighting, distinctive details. So something like: "The image captures Dover Castle perched on a steep, grassy hill in Kent, England, under a partly cloudy sky. Its round stone towers, crenellated walls, arched gatehouse and a small bridge are illuminated by bright daylight, highlighting the medieval stonework." That's two sentences. That includes context: Dover Castle, location, date/time? We could mention capture date/time: "Captured on 15 August 2026 at 15:59 UTC+01:00". But we need to incorporate relevant context. The description should combine relevant context with main visible subject, setting, action, lighting, and distinctive details. So we can mention the date/time and GPS as context, but not claim they are visible. So we can say "Captured on 15 August 2026 at 15:59 UTC+01:00, the photograph shows Dover Castle perched on a steep grassy hill in Kent, England, under a partly cloudy sky." That's one sentence. Then second sentence: "The stone fortress features round towers, crenellated battlements, an arched entrance, and a small bridge, all bathed in bright daylight that accentuates the historic stonework." That's two sentences. That's fine.

Now keywords: 10-18 unique, comma-separated keywords covering relevant context and visible details. Must be British English. Should include: Dover Castle, Dover, England, United Kingdom, Europe, Kent, hill, grassy, medieval, stone, fortress, round towers, crenellated, battlements, arched entrance, bridge, sky, partly cloudy, daylight, architecture, historic, Adobe Stock, Any Vision? The hints include "Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue". We need 10-18 unique keywords. Let's pick 14. Something like: Dover Castle, Kent, England, United Kingdom, Europe, medieval fortress, stone architecture, round towers, crenellated walls, arched gate
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
- *Observations:* Extra text appears before the Title field; Title has 4 words
  (requested 5-10); Keyword list has 44 terms (requested 10-18); Duplicate
  keywords: uk, europe, architecture
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 11.30s
- *Generation time:* 46.24s
- *Total time:* 57.57s
- *Input validation time:* 0.000134
- *Prompt preparation time:* 0.0121
- *First-token latency:* 6.13
- *Cleanup time:* 0.739
- *Prompt tokens:* 1,627
- *Generation tokens:* 1,000
- *Total tokens:* 2,627
- *Prompt throughput (raw):* 265 tok/s
- *Generation throughput (raw):* 29.1 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.17
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00118
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1538
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
Alright, let's get to work. I've got the image of Dover Castle in my sights, and I need to generate some metadata. First things first, I'll break down the request. It's a straightforward task: title, description, and keywords. The image is of Dover Castle, and I have the context: the date, time, and GPS coordinates. I'll use those as my foundation.

Now, the image itself. It's a classic view of the castle, a medieval fortress with those iconic round towers and the massive stone walls. The sky is partly cloudy, and the castle is perched on a grassy hill. I can see the arched entrance, and there's a small bridge over a moat or ditch. The lighting is bright, and the overall feel is historical and imposing.

I'll start with the title. "Dover Castle, England" is a good start, but I can do better. "Dover Castle on a Hill" is a bit too generic. "Dover Castle, Kent, England" is accurate, but I want something more evocative. "Dover Castle on a Steep Hill" is a good option. I'll go with "Dover Castle, Kent, England" as it's concise and accurate.

Next, the description. I need to weave in the context: the date, time, and location. I'll start with the main subject: "An exterior view of Dover Castle, a historic medieval fortress..." Then, I'll add the setting: "...perched on a steep grassy hill under a partly cloudy sky." I'll include the distinctive architectural details: "...featuring its iconic round towers and arched entranceway." And finally, I'll mention the bridge: "A small bridge crosses a ditch in the foreground." That should be a solid, factual description.

Finally, the keywords. I need to cover the subject, location, and key features. I'll start with the obvious: "Dover Castle, England, UK, Europe, Kent, Dover, Castle, Fortress, Architecture, Medieval, Stone, Tower, Hill, Wall, Hillfort, Ruin, Historic, Ancient, Defensive Wall, Partly Cloudy Sky, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Grassy Hill, Steep Hill, UK, United Kingdom, GBR, British, Europe, Architecture, History, Ancient History, Medieval History, Landmark, Tourist Attraction, Travel, Landscape, Scenery, Sky, Clouds, Damp, Green, Vegetation, Grass, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Hill, Steep Hill, Grass, Vegetation, Sky, Clouds, Damp, Green, Grass, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Hill, Steep Hill, Grass, Vegetation, Sky, Clouds, Damp, Green, Grass, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Hill, Steep Hill, Grass, Vegetation, Sky, Clouds, Damp, Green, Grass, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Hill, Steep Hill, Grass, Vegetation, Sky, Clouds, Damp, Green, Grass, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Hill, Steep Hill, Grass, Vegetation, Sky, Clouds, Damp, Green, Grass, Ditch, Moat, Bridge, Round
&lt;/think&gt;

Title: Dover Castle, Kent, England

Description: An exterior view of Dover Castle, a historic medieval fortress perched on a steep grassy hill under a partly cloudy sky. The iconic round towers and arched entranceway are visible, with a small bridge crossing a ditch in the foreground.

Keywords: Dover Castle, England, UK, Europe, Kent, Dover, Castle, Fortress, Architecture, Medieval, Stone, Tower, Hill, Wall, Hillfort, Ruin, Historic, Ancient, Defensive Wall, Partly Cloudy Sky, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Grassy Hill, Steep Hill, UK, United Kingdom, GBR, British, Europe, Architecture, History, Ancient History, Medieval History, Landmark, Tourist Attraction, Travel, Landscape, Scenery, Sky, Clouds,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get to work. I've got the image of Dover Castle in my sights, and I need to generate some metadata. First things first, I'll break down the request. It's a straightforward task: title, description, and keywords. The image is of Dover Castle, and I have the context: the date, time, and GPS coordinates. I'll use those as my foundation.

Now, the image itself. It's a classic view of the castle, a medieval fortress with those iconic round towers and the massive stone walls. The sky is partly cloudy, and the castle is perched on a grassy hill. I can see the arched entrance, and there's a small bridge over a moat or ditch. The lighting is bright, and the overall feel is historical and imposing.

I'll start with the title. "Dover Castle, England" is a good start, but I can do better. "Dover Castle on a Hill" is a bit too generic. "Dover Castle, Kent, England" is accurate, but I want something more evocative. "Dover Castle on a Steep Hill" is a good option. I'll go with "Dover Castle, Kent, England" as it's concise and accurate.

Next, the description. I need to weave in the context: the date, time, and location. I'll start with the main subject: "An exterior view of Dover Castle, a historic medieval fortress..." Then, I'll add the setting: "...perched on a steep grassy hill under a partly cloudy sky." I'll include the distinctive architectural details: "...featuring its iconic round towers and arched entranceway." And finally, I'll mention the bridge: "A small bridge crosses a ditch in the foreground." That should be a solid, factual description.

Finally, the keywords. I need to cover the subject, location, and key features. I'll start with the obvious: "Dover Castle, England, UK, Europe, Kent, Dover, Castle, Fortress, Architecture, Medieval, Stone, Tower, Hill, Wall, Hillfort, Ruin, Historic, Ancient, Defensive Wall, Partly Cloudy Sky, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Grassy Hill, Steep Hill, UK, United Kingdom, GBR, British, Europe, Architecture, History, Ancient History, Medieval History, Landmark, Tourist Attraction, Travel, Landscape, Scenery, Sky, Clouds, Damp, Green, Vegetation, Grass, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Hill, Steep Hill, Grass, Vegetation, Sky, Clouds, Damp, Green, Grass, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Hill, Steep Hill, Grass, Vegetation, Sky, Clouds, Damp, Green, Grass, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Hill, Steep Hill, Grass, Vegetation, Sky, Clouds, Damp, Green, Grass, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Hill, Steep Hill, Grass, Vegetation, Sky, Clouds, Damp, Green, Grass, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Hill, Steep Hill, Grass, Vegetation, Sky, Clouds, Damp, Green, Grass, Ditch, Moat, Bridge, Round
</think>

Title: Dover Castle, Kent, England

Description: An exterior view of Dover Castle, a historic medieval fortress perched on a steep grassy hill under a partly cloudy sky. The iconic round towers and arched entranceway are visible, with a small bridge crossing a ditch in the foreground.

Keywords: Dover Castle, England, UK, Europe, Kent, Dover, Castle, Fortress, Architecture, Medieval, Stone, Tower, Hill, Wall, Hillfort, Ruin, Historic, Ancient, Defensive Wall, Partly Cloudy Sky, Ditch, Moat, Bridge, Round Tower, Arched Entrance, Exterior View, Grassy Hill, Steep Hill, UK, United Kingdom, GBR, British, Europe, Architecture, History, Ancient History, Medieval History, Landmark, Tourist Attraction, Travel, Landscape, Scenery, Sky, Clouds,
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
- *Model load time:* 1.29s
- *Generation time:* 2.31s
- *Total time:* 3.61s
- *Input validation time:* 0.000213
- *Prompt preparation time:* 0.0029
- *First-token latency:* 0.109
- *Cleanup time:* 0.102
- *Prompt tokens:* 347
- *Generation tokens:* 78
- *Total tokens:* 425
- *Prompt throughput (raw):* 3,180 tok/s
- *Generation throughput (raw):* 290 tok/s
- *Peak memory:* 2.1
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.00149
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1443
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
Dover Castle, England, UK, GBR, Europe, Medieval, Round Towers, Arched Entranceway, Small Bridge, steep hill, partly cloudy sky, Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue.
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
- *Observations:* Response repeats the same text; Missing or empty fields:
  Title; Extra text appears before the Title field; Response appears cut off
  at the token limit; Internal reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.78s
- *Generation time:* 40.83s
- *Total time:* 42.63s
- *Input validation time:* 8.54e-05
- *Prompt preparation time:* 0.00205
- *First-token latency:* 10.7
- *Cleanup time:* 0.207
- *Prompt tokens:* 6,344
- *Generation tokens:* 1,000
- *Total tokens:* 7,344
- *Prompt throughput (raw):* 592 tok/s
- *Generation throughput (raw):* 34.0 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.43
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1405
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
&lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The image is of Dover Castle, so maybe "Dover Castle Exterior View". Then the description should combine context (date/time, location) with the main subject, setting, etc. The image shows a historic medieval stone castle with round towers, arched entrance, steep grassy hill, partly cloudy sky. So the description should mention the castle, its features, the setting, and the time. Then keywords need to be 10-18 unique ones, covering context and visible details.

First, title: Let's make it "Dover Castle Exterior with Round Towers". Wait, need 5-10 words. Maybe "Dover Castle Medieval Exterior". Wait, let's check the hints. The title hint is Dover Castle, Dover, England... so maybe "Dover Castle Exterior View". That's 3 words, maybe add more. "Dover Castle Historic Exterior with Round Towers" – that's 6 words. Okay.

Description: Combine context (capture date/time, location) with main subject (historic medieval stone castle), setting (steep grassy hill), lighting (partly cloudy sky), distinctive details (round towers, arched entrance). So: "A historic medieval stone castle with round towers and an arched entrance, situated on a steep grassy hill under a partly cloudy sky, captured on 2026-08-15 at 15:59:46 UTC+01:00 in Dover, England." Wait, need 1-2 sentences. Let's make it concise. "A historic medieval stone castle with round towers and an arched entrance, built on a steep grassy hill under a partly cloudy sky, photographed in Dover, England on 2026-08-15." That's two sentences? Wait, no, 1-2 sentences. Let's try: "A historic medieval stone castle with round towers and an arched entrance, situated on a steep grassy hill under a partly cloudy sky, captured in Dover, England on 2026-08-15." That's one sentence. Wait, need to include the context (capture date/time, location) and the main visible subject, setting, etc. So maybe: "A historic medieval stone castle with round towers and an arched entrance, built on a steep grassy hill under a partly cloudy sky, photographed in Dover, England on 2026-08-15 15:59:46 UTC+01:00." That's a bit long, but maybe shorten. "A historic medieval stone castle with round towers and an arched entrance, situated on a steep grassy hill under a partly cloudy sky, captured in Dover, England on 2026-08-15." That works.

Keywords: Need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (but maybe not, since the image might not be from those, but the hints include them? Wait, the hints say "Adobe Stock, Any Vision" as keyword hints? Wait, the context says "Descriptive hints" include those, but maybe the keywords should include the ones from hints plus others. Let's list: Dover Castle, England, Europe, Kent, Castle, Fortress, Hill, Sky, Stone, Tower, Wall, ancient, architecture, blue, Britain, UK, United Kingdom, Arch, Bridge. Wait, the image has a small bridge, so Bridge. Let's check the hints: keyword hints are Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue. So combine these, remove duplicates. Let's list unique ones: Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue. Wait, need 10-18. Let's count: Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue – that's 18, but maybe some can be combined. Wait, "United Kingdom" and "UK" are similar, but keep both? Or pick one. Let's do: Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, Wall, ancient, architecture, blue. That's 17. Add Bridge if visible. The image has a small bridge, so Bridge. So Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, Wall, ancient, architecture, blue, Bridge. That's
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The image is of Dover Castle, so maybe "Dover Castle Exterior View". Then the description should combine context (date/time, location) with the main subject, setting, etc. The image shows a historic medieval stone castle with round towers, arched entrance, steep grassy hill, partly cloudy sky. So the description should mention the castle, its features, the setting, and the time. Then keywords need to be 10-18 unique ones, covering context and visible details.

First, title: Let's make it "Dover Castle Exterior with Round Towers". Wait, need 5-10 words. Maybe "Dover Castle Medieval Exterior". Wait, let's check the hints. The title hint is Dover Castle, Dover, England... so maybe "Dover Castle Exterior View". That's 3 words, maybe add more. "Dover Castle Historic Exterior with Round Towers" – that's 6 words. Okay.

Description: Combine context (capture date/time, location) with main subject (historic medieval stone castle), setting (steep grassy hill), lighting (partly cloudy sky), distinctive details (round towers, arched entrance). So: "A historic medieval stone castle with round towers and an arched entrance, situated on a steep grassy hill under a partly cloudy sky, captured on 2026-08-15 at 15:59:46 UTC+01:00 in Dover, England." Wait, need 1-2 sentences. Let's make it concise. "A historic medieval stone castle with round towers and an arched entrance, built on a steep grassy hill under a partly cloudy sky, photographed in Dover, England on 2026-08-15." That's two sentences? Wait, no, 1-2 sentences. Let's try: "A historic medieval stone castle with round towers and an arched entrance, situated on a steep grassy hill under a partly cloudy sky, captured in Dover, England on 2026-08-15." That's one sentence. Wait, need to include the context (capture date/time, location) and the main visible subject, setting, etc. So maybe: "A historic medieval stone castle with round towers and an arched entrance, built on a steep grassy hill under a partly cloudy sky, photographed in Dover, England on 2026-08-15 15:59:46 UTC+01:00." That's a bit long, but maybe shorten. "A historic medieval stone castle with round towers and an arched entrance, situated on a steep grassy hill under a partly cloudy sky, captured in Dover, England on 2026-08-15." That works.

Keywords: Need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (but maybe not, since the image might not be from those, but the hints include them? Wait, the hints say "Adobe Stock, Any Vision" as keyword hints? Wait, the context says "Descriptive hints" include those, but maybe the keywords should include the ones from hints plus others. Let's list: Dover Castle, England, Europe, Kent, Castle, Fortress, Hill, Sky, Stone, Tower, Wall, ancient, architecture, blue, Britain, UK, United Kingdom, Arch, Bridge. Wait, the image has a small bridge, so Bridge. Let's check the hints: keyword hints are Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue. So combine these, remove duplicates. Let's list unique ones: Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue. Wait, need 10-18. Let's count: Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue – that's 18, but maybe some can be combined. Wait, "United Kingdom" and "UK" are similar, but keep both? Or pick one. Let's do: Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, Wall, ancient, architecture, blue. That's 17. Add Bridge if visible. The image has a small bridge, so Bridge. So Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, Wall, ancient, architecture, blue, Bridge. That's
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
- *Observations:* Response repeats the same text; Missing or empty fields:
  Title; Extra text appears before the Title field; Conversation-role control
  tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 3.57s
- *Generation time:* 185.63s
- *Total time:* 189.23s
- *Input validation time:* 0.000424
- *Prompt preparation time:* 0.0025
- *First-token latency:* 1.77
- *Cleanup time:* 0.515
- *Prompt tokens:* 1,296
- *Generation tokens:* 834
- *Total tokens:* 2,130
- *Prompt throughput (raw):* 730 tok/s
- *Generation throughput (raw):* 4.56 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.8
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00365
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1535
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
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords.

Starting with the title: The image shows Dover Castle, which is in the context given. The castle is a historic medieval stone structure with round towers, an arched entrance, and a bridge. The setting is a grassy hill under a partly cloudy sky. So the title should be concise. Maybe "Dover Castle, Kent, UK: Medieval stone fortress on grassy hill with arched entrance and towers." That's 10 words.

Next, the description: It needs to combine context and visible details. The capture date and GPS are authoritative but not visible, so focus on the image. The castle is on a grassy hill, has stone walls, round towers, an arched entrance, and a bridge. The sky is partly cloudy. So the description could be: "Exterior view of Dover Castle, a medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a bridge, under a partly cloudy sky." That's 25 words, which is a bit long but fits the requirement.

Keywords: Need 10-18 unique terms. From the context and image: Dover Castle, Kent, UK, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, blue (sky), United Kingdom, Britain, fortification, wall. Let me count: Dover Castle, Kent, UK, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, blue, United Kingdom, Britain, fortification, wall. That's 17 keywords. Need to check if they are unique and relevant. Maybe "UK" and "United Kingdom" are redundant, so replace one. Let's remove "United Kingdom" and keep UK. Also, "Britain" is similar to UK. So revised: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, blue sky, Kent, England. Wait, Kent is already in the title. Maybe include "Dover Castle" as a keyword. Let me check again. The context mentions Dover Castle, so that's essential. So final keywords: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, blue sky, UK. That's 16 keywords. Maybe add "sky" instead of "blue sky" but "blue" is in the hints. Hmm, the hints have "blue" as a keyword. So include that. So final keywords: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, blue, UK. That's 16. Maybe add "sky" instead of "blue" if needed, but the hint says to prefer image evidence. The sky is partly cloudy, so "partly cloudy sky" is better. So keep "partly cloudy sky". Alright, that should work.◁/think▷Title: Dover Castle, Kent, UK: Medieval stone fortress on grassy hill with arched entrance and towers
Description: Exterior view of Dover Castle, a medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a bridge, under a partly cloudy sky.
Keywords: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, UK&lt;|im_assistant|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords. 

Starting with the title: The image shows Dover Castle, which is in the context given. The castle is a historic medieval stone structure with round towers, an arched entrance, and a bridge. The setting is a grassy hill under a partly cloudy sky. So the title should be concise. Maybe "Dover Castle, Kent, UK: Medieval stone fortress on grassy hill with arched entrance and towers." That's 10 words.

Next, the description: It needs to combine context and visible details. The capture date and GPS are authoritative but not visible, so focus on the image. The castle is on a grassy hill, has stone walls, round towers, an arched entrance, and a bridge. The sky is partly cloudy. So the description could be: "Exterior view of Dover Castle, a medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a bridge, under a partly cloudy sky." That's 25 words, which is a bit long but fits the requirement.

Keywords: Need 10-18 unique terms. From the context and image: Dover Castle, Kent, UK, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, blue (sky), United Kingdom, Britain, fortification, wall. Let me count: Dover Castle, Kent, UK, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, blue, United Kingdom, Britain, fortification, wall. That's 17 keywords. Need to check if they are unique and relevant. Maybe "UK" and "United Kingdom" are redundant, so replace one. Let's remove "United Kingdom" and keep UK. Also, "Britain" is similar to UK. So revised: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, blue sky, Kent, England. Wait, Kent is already in the title. Maybe include "Dover Castle" as a keyword. Let me check again. The context mentions Dover Castle, so that's essential. So final keywords: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, blue sky, UK. That's 16 keywords. Maybe add "sky" instead of "blue sky" but "blue" is in the hints. Hmm, the hints have "blue" as a keyword. So include that. So final keywords: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, blue, UK. That's 16. Maybe add "sky" instead of "blue" if needed, but the hint says to prefer image evidence. The sky is partly cloudy, so "partly cloudy sky" is better. So keep "partly cloudy sky". Alright, that should work.◁/think▷Title: Dover Castle, Kent, UK: Medieval stone fortress on grassy hill with arched entrance and towers  
Description: Exterior view of Dover Castle, a medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a bridge, under a partly cloudy sky.  
Keywords: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, UK<|im_assistant|>
```

</details>

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### mlx-community/LFM2.5-VL-1.6B-bf16

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Title has 13 words (requested 5-10); Keyword list has 454
  terms (requested 10-18); Duplicate keywords: uk, architecture, blue, castle
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.81s
- *Generation time:* 7.07s
- *Total time:* 7.89s
- *Input validation time:* 0.000641
- *Prompt preparation time:* 0.00625
- *First-token latency:* 0.114
- *Cleanup time:* 0.215
- *Prompt tokens:* 603
- *Generation tokens:* 1,000
- *Total tokens:* 1,603
- *Prompt throughput (raw):* 5,300 tok/s
- *Generation throughput (raw):* 164 tok/s
- *Peak memory:* 4.1
- *Active memory:* 3.2
- *Cache memory:* 0.03
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.00379
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1399
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
Title: Dover Castle, England, UK, Europe, Fortress, Medieval, Stone, Tower, Hill, Sky, Architecture, Blue

Description: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky. The castle's imposing structure and intricate design reflect its rich history and architectural significance.

Keywords: Dover Castle, England, UK, Europe, Fortress, Medieval, Stone, Tower, Hill, Sky, Architecture, Blue, Castle, Kent, UK, Ancient, Blue, Architecture, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle,
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
- *Observations:* Missing or empty fields: Title, Description; Extra text
  appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Model load time:* 0.94s
- *Generation time:* 1.25s
- *Total time:* 2.21s
- *Input validation time:* 8.74e-05
- *Prompt preparation time:* 0.0118
- *First-token latency:* 0.277
- *Cleanup time:* 0.106
- *Prompt tokens:* 1,004
- *Generation tokens:* 41
- *Total tokens:* 1,045
- *Prompt throughput (raw):* 3,629 tok/s
- *Generation throughput (raw):* 247 tok/s
- *Peak memory:* 3.7
- *Active memory:* 2.3
- *Cache memory:* 0.06
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00483
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1416
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
Ancient stone castle on hillside under clear sky.

Keywords: castle, stone, hill, medieval, arch, tower, bridge, UK, Europe, grassy, sky, fortress, architecture
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
- *Model load time:* 2.27s
- *Generation time:* 21.95s
- *Total time:* 24.23s
- *Input validation time:* 0.000169
- *Prompt preparation time:* 0.0052
- *First-token latency:* 2.89
- *Cleanup time:* 0.185
- *Prompt tokens:* 3,158
- *Generation tokens:* 113
- *Total tokens:* 3,271
- *Prompt throughput (raw):* 1,091 tok/s
- *Generation throughput (raw):* 6.27 tok/s
- *Peak memory:* 24
- *Active memory:* 19
- *Cache memory:* 5.0
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 0.00655
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1386
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
Dover Castle: Medieval Stone Fortress on Steep Hillside

A historic stone castle with round towers and arched entranceway stands atop a steep, grassy hill in Dover, England. Built of brown and white stones, the fortress features a small bridge and is set against a partly cloudy blue sky, with sunlight casting shadows on the hillside.

Dover Castle, England, UK, Europe, Stone, Fortress, Hill, Kent, Sky, Tower, Arch, Bridge, Medieval, Castle, Wall, Ancient, Architecture, Blue, Sunlight, Shadows
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
- *Observations:* Response repeats the same text; Title has 3 words (requested
  5-10); Keyword list has 33 terms (requested 10-18); Duplicate keywords:
  england, uk, europe, fortress, castle, stone, tower, wall, hill, sky,
  ancient, architecture, united kingdom
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.76s
- *Generation time:* 31.90s
- *Total time:* 32.66s
- *Input validation time:* 0.000115
- *Prompt preparation time:* 0.000339
- *First-token latency:* 29.5
- *Cleanup time:* 0.0893
- *Prompt tokens:* 16,506
- *Generation tokens:* 118
- *Total tokens:* 16,624
- *Prompt throughput (raw):* 559 tok/s
- *Generation throughput (raw):* 92.1 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00775
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1420
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
Dover Castle Exterior

Description:
An exterior view of Dover Castle, a historic medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a small bridge under a partly cloudy sky.

Keywords: Dover Castle, England, UK, Europe, Fortress, Castle, Stone, Tower, Wall, Hill, Sky, Arch, Ancient, Architecture, Britain, Kent, United Kingdom, Stone, Wall, Castle, England, Europe, Fortress, Hill, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
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
- *Observations:* Extra text appears before the Title field; Title has 11
  words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.75s
- *Generation time:* 37.39s
- *Total time:* 38.17s
- *Input validation time:* 7.51e-05
- *Prompt preparation time:* 0.0112
- *First-token latency:* 25.2
- *Cleanup time:* 0.142
- *Prompt tokens:* 16,508
- *Generation tokens:* 898
- *Total tokens:* 17,406
- *Prompt throughput (raw):* 654 tok/s
- *Generation throughput (raw):* 80.9 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.7
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00806
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1428
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
Got it, let's tackle this step by step. First, the title. The context says Dover Castle, so the title should be specific. The hint says Dover Castle, so "Dover Castle, England" is good. Need 5-10 words. Let's see: "Dover Castle, England, Historic Medieval Stone Castle on Hill" – that's 10 words. Wait, maybe "Dover Castle, England: Exterior of Historic Medieval Stone Castle" – but need to check if the image matches. The image shows a castle with round towers, arch, bridge, hill. So title should be precise.

Next, description. Need 1-2 sentences. Context has capture date, GPS, but we don't claim those. The main subject is the castle. Visible details: exterior view, round towers, arched entrance, small bridge, steep grassy hill, partly cloudy sky. So: "An exterior view of Dover Castle, England, showcasing its historic stone construction, round towers, arched entranceway, and a small bridge atop a steep grassy hill under a partly cloudy sky."

Then keywords. Need 10-18 unique, comma-separated. From context: Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue. But check the image: the castle is stone, round towers, arch, bridge, hill, sky. So include those. Let's list: Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture. Wait, need to pick 10-18. Let's see: Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture. That's 15. Let's check if all are visible. The image has stone, towers, arch, hill, sky, blue sky (partly cloudy, so blue is visible). So keywords: Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture. Wait, but the context has "Britain" and "UK", so maybe include those. Let's make sure they are unique and cover context and visible details.

Wait, the descriptive hints have "Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue". So need to pick from those, but not repeat. Let's list:

Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture.

Wait, "Wall" is in the context but not visible. The image has a wall (the castle walls), so "Wall" is a keyword. Let's check: the castle has walls, so "Wall" is good. So include Wall. Let's see:

Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture, Wall.

That's 16. Let's check if all are visible. The image has a castle (Castle), stone walls (Stone), towers (Tower), arch (Arch), hill (Hill), sky (Sky), blue (partly cloudy, so blue is visible), Fortress (since it's a fort), Kent (the location is Kent, England), UK, Ancient (the castle is ancient), Wall (the walls), and Dover Castle (title). So that's good.

Now, title: "Dover Castle, England: Exterior of Historic Medieval Stone Castle on Hill" –
&lt;/think&gt;

Title: Dover Castle, England: Exterior of Historic Medieval Stone Castle on Hill

Description: An exterior view of Dover Castle, England, showcasing its historic stone construction, round towers, arched entranceway, and a small bridge atop a steep grassy hill under a partly cloudy sky.

Keywords: Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture, Wall
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this step by step. First, the title. The context says Dover Castle, so the title should be specific. The hint says Dover Castle, so "Dover Castle, England" is good. Need 5-10 words. Let's see: "Dover Castle, England, Historic Medieval Stone Castle on Hill" – that's 10 words. Wait, maybe "Dover Castle, England: Exterior of Historic Medieval Stone Castle" – but need to check if the image matches. The image shows a castle with round towers, arch, bridge, hill. So title should be precise.

Next, description. Need 1-2 sentences. Context has capture date, GPS, but we don't claim those. The main subject is the castle. Visible details: exterior view, round towers, arched entrance, small bridge, steep grassy hill, partly cloudy sky. So: "An exterior view of Dover Castle, England, showcasing its historic stone construction, round towers, arched entranceway, and a small bridge atop a steep grassy hill under a partly cloudy sky."

Then keywords. Need 10-18 unique, comma-separated. From context: Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue. But check the image: the castle is stone, round towers, arch, bridge, hill, sky. So include those. Let's list: Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture. Wait, need to pick 10-18. Let's see: Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture. That's 15. Let's check if all are visible. The image has stone, towers, arch, hill, sky, blue sky (partly cloudy, so blue is visible). So keywords: Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture. Wait, but the context has "Britain" and "UK", so maybe include those. Let's make sure they are unique and cover context and visible details.

Wait, the descriptive hints have "Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue". So need to pick from those, but not repeat. Let's list:

Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture.

Wait, "Wall" is in the context but not visible. The image has a wall (the castle walls), so "Wall" is a keyword. Let's check: the castle has walls, so "Wall" is good. So include Wall. Let's see:

Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture, Wall.

That's 16. Let's check if all are visible. The image has a castle (Castle), stone walls (Stone), towers (Tower), arch (Arch), hill (Hill), sky (Sky), blue (partly cloudy, so blue is visible), Fortress (since it's a fort), Kent (the location is Kent, England), UK, Ancient (the castle is ancient), Wall (the walls), and Dover Castle (title). So that's good.

Now, title: "Dover Castle, England: Exterior of Historic Medieval Stone Castle on Hill" – 
</think>

Title: Dover Castle, England: Exterior of Historic Medieval Stone Castle on Hill

Description: An exterior view of Dover Castle, England, showcasing its historic stone construction, round towers, arched entranceway, and a small bridge atop a steep grassy hill under a partly cloudy sky.

Keywords: Dover Castle, England, Historic, Medieval, Stone, Castle, Tower, Arch, Hill, Sky, Blue, Fortress, Kent, UK, Ancient, Architecture, Wall
```

</details>

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
  the token limit; Title has 4 words (requested 5-10); Keyword list has 357
  terms (requested 10-18); Duplicate keywords: kent, england, uk, europe,
  medieval, stone, round towers, small bridge, grassy hill, partly cloudy sky,
  architecture, ancient, arched entranceway, historic, fortress
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.18s
- *Generation time:* 48.02s
- *Total time:* 49.21s
- *Input validation time:* 8.62e-05
- *Prompt preparation time:* 0.000483
- *First-token latency:* 25.1
- *Cleanup time:* 0.166
- *Prompt tokens:* 16,517
- *Generation tokens:* 1,000
- *Total tokens:* 17,517
- *Prompt throughput (raw):* 657 tok/s
- *Generation throughput (raw):* 45.8 tok/s
- *Peak memory:* 13
- *Active memory:* 9.5
- *Cache memory:* 1.3
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1478
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
Dover Castle, Kent, England

Description:
A historic medieval stone castle with round towers, an arched entranceway, and a small bridge, situated on a steep grassy hill under a partly cloudy sky, captured on 2026-08-15 at 15:59:46 UTC+01:00.

Keywords: Dover Castle, Kent, England, UK, Europe, medieval, stone, round towers, arched entrance, small bridge, grassy hill, partly cloudy sky, historic architecture, ancient fortress, stone walls, medieval castle, United Kingdom, Europe, architecture, blue sky, greenery, stone bridge, medieval structure, stone towers, historical landmark, Kent, England, UK, Europe, ancient, architecture, blue, grass, sky, stone, tower, wall, entrance, bridge, castle, Dover, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent
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
- *Model load time:* 2.35s
- *Generation time:* 7.54s
- *Total time:* 9.93s
- *Input validation time:* 9.41e-05
- *Prompt preparation time:* 0.00278
- *First-token latency:* 0.381
- *Cleanup time:* 0.292
- *Prompt tokens:* 607
- *Generation tokens:* 261
- *Total tokens:* 868
- *Prompt throughput (raw):* 1,595 tok/s
- *Generation throughput (raw):* 40.2 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0113
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1405
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
&#96;&#96;&#96;json
{
  "context": {
    "title": "Dover Castle",
    "location": "Dover, England, UK, GBR, Europe",
    "description": "An exterior view of a historic medieval stone castle, featuring round towers, an arched entrance, and a steep grassy hill under a partly cloudy sky."
  },
  "date_time": "2026-08-15 15:59:46 UTC+01:00",
  "gps": "51.128800°N, 1.319100°E",
  "keywords": [
    "Adobe Stock",
    "Aerial View",
    "Arch",
    "Britain",
    "Castle",
    "Dover Castle",
    "England",
    "Europe",
    "Fortress",
    "Hill",
    "Kent",
    "Sky",
    "Stone",
    "Tower",
    "UK",
    "United Kingdom",
    "Wall",
    "Ancient",
    "Architecture",
    "Blue"
  ]
}
&#96;&#96;&#96;
</pre>

<details>
<summary>Exact raw output</summary>

````text
```json
{
  "context": {
    "title": "Dover Castle",
    "location": "Dover, England, UK, GBR, Europe",
    "description": "An exterior view of a historic medieval stone castle, featuring round towers, an arched entrance, and a steep grassy hill under a partly cloudy sky."
  },
  "date_time": "2026-08-15 15:59:46 UTC+01:00",
  "gps": "51.128800°N, 1.319100°E",
  "keywords": [
    "Adobe Stock",
    "Aerial View",
    "Arch",
    "Britain",
    "Castle",
    "Dover Castle",
    "England",
    "Europe",
    "Fortress",
    "Hill",
    "Kent",
    "Sky",
    "Stone",
    "Tower",
    "UK",
    "United Kingdom",
    "Wall",
    "Ancient",
    "Architecture",
    "Blue"
  ]
}
```
````

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
- *Model load time:* 0.97s
- *Generation time:* 5.16s
- *Total time:* 6.14s
- *Input validation time:* 0.000113
- *Prompt preparation time:* 0.00217
- *First-token latency:* 3.27
- *Cleanup time:* 0.115
- *Prompt tokens:* 2,734
- *Generation tokens:* 53
- *Total tokens:* 2,787
- *Prompt throughput (raw):* 836 tok/s
- *Generation throughput (raw):* 57.3 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.90
- *Model-load active memory:* 8.05
- *Post-cleanup active memory:* 0.0124
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1350
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
Dover Castle, Dover, England, UK, GBR, Europe, historic, medieval, stone, castle, round towers, arched entranceway, steep grassy hill, partly cloudy sky, fortress, ancient, architecture, blue
</pre>

<details>
<summary>Exact raw output</summary>

```text
Dover Castle, Dover, England, UK, GBR, Europe, historic, medieval, stone, castle, round towers, arched entranceway, steep grassy hill, partly cloudy sky, fortress, ancient, architecture, blue 
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
- *Model load time:* 0.79s
- *Generation time:* 1.74s
- *Total time:* 2.54s
- *Input validation time:* 9.73e-05
- *Prompt preparation time:* 0.00258
- *First-token latency:* 0.0877
- *Cleanup time:* 0.126
- *Prompt tokens:* 343
- *Generation tokens:* 81
- *Total tokens:* 424
- *Prompt throughput (raw):* 3,909 tok/s
- *Generation throughput (raw):* 186 tok/s
- *Peak memory:* 2.1
- *Active memory:* 0.61
- *Cache memory:* 0.28
- *Model-load active memory:* 0.611
- *Post-cleanup active memory:* 0.013
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1434
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
Title: Dover Castle, England, UK, GBR, Europe
Description: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky. The castle is situated on a hill under a partly cloudy sky, and it is a part of the historic medieval architecture in England.
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
  of only returning the requested fields
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Model load time:* 1.42s
- *Generation time:* 25.38s
- *Total time:* 26.82s
- *Input validation time:* 0.000109
- *Prompt preparation time:* 0.000251
- *First-token latency:* 1.63
- *Cleanup time:* 0.226
- *Prompt tokens:* 4,435
- *Generation tokens:* 726
- *Total tokens:* 5,161
- *Prompt throughput (raw):* 2,719 tok/s
- *Generation throughput (raw):* 32.1 tok/s
- *Peak memory:* 4.3
- *Active memory:* 1.7
- *Cache memory:* 2.0
- *Model-load active memory:* 1.73
- *Post-cleanup active memory:* 0.014
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1334
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
- Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
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
  mlx-community/SmolVLM2-2.2B-Instruct-mlx: Image features and image tokens do
  not match: tokens: 81, features 1053
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* Image features and image tokens do not match:
  tokens: 81, features 1053
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.82s
- *Generation time:* 2.11s
- *Total time:* 2.94s
- *Input validation time:* 0.000169
- *Prompt preparation time:* 0.00511
- *First-token latency:* -
- *Cleanup time:* 0.122
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
- *Rendered prompt characters:* 1380
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11966, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12315, in _generate_once
    return strict_generate(
        model=model,
    ...<3 lines>...
        **generate_kwargs,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1159, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 978, in stream_generate
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
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/idefics3/idefics3.py", line 174, in _prepare_inputs_for_multimodal
    raise ValueError(
        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
    )
ValueError: Image features and image tokens do not match: tokens: 81, features 1053

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12893, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12328, in _run_model_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11973, in _run_generation_guarded
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: Image features and image tokens do not match: tokens: 81, features 1053

```

#### Captured upstream output

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '8', '1', '5', '-', '1', '5', '5', '9', '4', '6', '_', 'D', 'S', 'C', '0', '1', '5', '6', '8', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>User:<image>Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-15 15:59:46 UTC+01:00
- GPS: 51.128800°N, 1.319100°E

Descriptive hints:
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue

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
Fetching 12 files: 100%|##########| 12/12 [00:00<00:00, 3113.43it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[00:45:18] ERROR    Generation error for mlx-community/SmolVLM2-2.2B-Instruct-mlx
                    ValueError: Image features and image tokens do not match: tokens: 81, features 1053
```

</details>

---
