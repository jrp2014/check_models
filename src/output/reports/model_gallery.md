# Model Output Gallery

Generated on: 2026-07-18 19:23:21 BST

Complete per-model evidence artifact with image metadata, the source prompt,
summary tables, diagnostics, and full generated output for every attempted
model.

_Action Snapshot: see [results.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.md) for the full summary._

## Run Stamps

- `mlx-vlm`: `0.6.5`
- `mlx`: `0.32.1.dev20260718+b7c3dd6d`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.24.0`
- _Python Version:_ 3.13.13
- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.2
- _GPU/Chip:_ Apple M5 Max

## Image Metadata

- _Date:_ 2026-04-25 23:49:44 BST
- _Time:_ 23:49:44

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Model Output and Cost Summary

Every model in this run, with complete successful output or a concise failure diagnostic beside runtime, memory, and quality signals.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Result                        | Output / diagnostic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Gen tok   | Total   | Gen TPS   | Peak GB   | Quality signal                       |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|---------|-----------|-----------|--------------------------------------|
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `avoid` / `runtime failure`   | [mlx-vlm; model-error] Error: Model Error - Model loading failed: Server disconnected without sending a response.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | -         | 33.13s  | -         | -         | mlx-vlm; model-error                 |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `avoid` / `runtime failure`   | [mlx-vlm; model-error] Error: Model Error - Model loading failed: Server disconnected without sending a response.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | -         | 30.54s  | -         | -         | mlx-vlm; model-error                 |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `avoid` / `runtime failure`   | [mlx-vlm; model-error] Error: Model Error - Model loading failed: Server disconnected without sending a response.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | -         | 45.00s  | -         | -         | mlx-vlm; model-error                 |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `avoid` / `runtime failure`   | [mlx-vlm; model-error] Error: Model Error - Model loading failed: Server disconnected without sending a response.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | -         | 30.46s  | -         | -         | mlx-vlm; model-error                 |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `avoid` / `runtime failure`   | [mlx-vlm; model-error] Error: Model Error - Model loading failed: Server disconnected without sending a response.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | -         | 51.47s  | -         | -         | mlx-vlm; model-error                 |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `avoid` / `runtime failure`   | [mlx-vlm; model-error] Error: Model Error - Model loading failed: Server disconnected without sending a response.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | -         | 30.70s  | -         | -         | mlx-vlm; model-error                 |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `clean-triage-pass` / `clean` | Two cats are laying on a pink couch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 10        | 31.14s  | 504       | 1.0       | clean                                |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `clean-triage-pass` / `clean` | Two cats are sleeping on a pink blanket.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 10        | 4.80s   | 313       | 3.0       | clean                                |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `clean-triage-pass` / `clean` | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 13        | 2.53s   | 188       | 4.1       | clean                                |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `clean-triage-pass` / `clean` | The image shows two cats lying on a pink blanket, with remote controls nearby.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 17        | 2.84s   | 259       | 3.0       | clean                                |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `clean-triage-pass` / `clean` | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats have striped coats, and they are positioned on a couch. The cats are facing the camera, and their eyes are open, indicating alertness. The image is a close-up, color photograph, and it captures the relaxed posture of the cats. There are no texts or other objects in the image.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | 81        | 1.24s   | 369       | 1.8       | clean                                |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `clean-triage-pass` / `clean` | The image shows two cats lying on a pink blanket. One cat is on the left side, while the other is on the right side. There are two remote controls placed on the blanket, one on the left side and the other on the right side.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | 52        | 2.02s   | 325       | 2.5       | clean                                |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `clean-triage-pass` / `clean` | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 35        | 1.14s   | 116       | 4.0       | clean                                |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `clean-triage-pass` / `clean` | Two tabby cats are lying on a pink blanket on a red couch, with two remote controls nearby.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 21        | 5.98s   | 128       | 16        | clean                                |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `clean-triage-pass` / `clean` | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 19        | 5.86s   | 60.4      | 9.2       | clean                                |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `clean-triage-pass` / `clean` | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 19        | 15.04s  | 60.7      | 9.2       | clean                                |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `clean-triage-pass` / `clean` | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 13        | 54.11s  | 121       | 5.5       | clean                                |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `clean-triage-pass` / `clean` | This image depicts two cats lying on a soft, pink cushion or blanket. - The cat on the left appears to be a kitten, with a smaller size, striped tabby pattern, and seems relaxed. - The cat on the right is an adult tabby cat, also striped, and is lying on its side with its head slightly raised, looking calm and content. Both are positioned near what looks like a remote control, suggesting a cozy, indoor setting.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 97        | 2.60s   | 196       | 4.5       | clean                                |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `clean-triage-pass` / `clean` | This is a close-up photograph of two cats sleeping on a bright pink couch. The cat on the left is a tabby with a black and grey pattern, lying on its back with its eyes closed. The cat on the right is a calico with a brown and black striped pattern, also lying on its back. Both cats are resting peacefully. There are two remote controls on the couch, one white and one with a blue button.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 89        | 4.10s   | 135       | 5.3       | clean                                |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `clean-triage-pass` / `clean` | This is a close-up photograph of two cats sleeping on a bright pink couch. The cat on the left is a tabby with a black and grey pattern, lying on its back with its eyes closed. The cat on the right is a calico with a brown and black striped pattern, also lying on its back. Both cats are resting peacefully. There are two remote controls on the couch, one white and one with a blue button.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 89        | 47.69s  | 134       | 5.1       | clean                                |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `clean-triage-pass` / `clean` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled with its head down. Two remote controls are also visible on the couch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 42        | 5.47s   | 89.3      | 7.7       | clean                                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `clean-triage-pass` / `clean` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls. The cat on the left is curled with its tail tucked, while the one on the right lies stretched out, both appearing deeply relaxed and comfortable.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 51        | 8.79s   | 117       | 21        | clean                                |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `clean-triage-pass` / `clean` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray. The cat on the left is curled with its tail tucked, while the one on the right lies stretched out, both appearing deeply relaxed in a cozy, domestic scene.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 61        | 5.95s   | 98.5      | 30        | clean                                |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `caveat` / `context budget`   | Cat.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 3         | 2.60s   | 82.3      | 4.6       | harness:long-context; context-budget |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `clean-triage-pass` / `clean` | The image shows two tabby cats lying on a pink couch. Both cats are stretched out, appearing relaxed. Between them are two remote controls. The couch has a vibrant pink cover, and the cats' fur has a mix of brown, black, and white stripes. The scene conveys a cozy and calm atmosphere.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 65        | 26.81s  | 65.7      | 10        | clean                                |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `clean-triage-pass` / `clean` | <\|channel>thought <channel\|>A high-angle shot shows two tabby cats sleeping on a pink blanket next to two remote controls.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 25        | 6.48s   | 27.9      | 29        | formatting                           |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `clean-triage-pass` / `clean` | <\|channel>thought <channel\|>An overhead shot shows two tabby cats sleeping on a pink blanket next to two remote controls.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 23        | 6.45s   | 24.9      | 28        | formatting                           |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `harness`           | So, let's see. The image shows two cats lying on a pink couch. The couch is a bright pink color, and there are two remote controls next to the cats. One cat is on the left, with a striped pattern, and the other is on the right, also striped but maybe a bit different. Both cats are relaxed, lying down, maybe sleeping. The remotes are white with some buttons, and one has a blue button. The background is the pink couch, so the main elements are the two cats, the remotes, and the pink fabric. I need to describe this briefly. &lt;/think&gt; Two tabby cats are lying on a bright pink couch, relaxed and asleep. A white remote control with a blue button sits between them, and another remote control is positioned near the left cat. The scene is cozy, with the cats' striped fur and the vibrant pink fabric of the couch forming the backdrop.                                                              | 188       | 4.51s   | 135       | 5.3       | harness:stop-token                   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `clean-triage-pass` / `clean` | This image shows two tabby cats lounging on a pink cushion. Both cats appear relaxed, lying on their sides and stretching out their paws. One cat faces left, while the other faces right, and there's a remote control near each of them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 54        | 3.15s   | 64.1      | 10        | clean                                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `clean-triage-pass` / `clean` | This image shows two cats lying on their backs on a pink blanket or cushion. Both cats appear relaxed and are positioned close to each other, with their paws in the air. There are two remote controls placed near them, suggesting a cozy indoor setting, possibly on a couch or bed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 60        | 3.44s   | 67.5      | 9.8       | clean                                |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `clean-triage-pass` / `clean` | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 25        | 5.31s   | 28.7      | 19        | clean                                |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `clean-triage-pass` / `clean` | In this image we can see two cats on the sofa. There are two remotes on the sofa.<end_of_utterance>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 23        | 33.07s  | 34.2      | 19        | formatting                           |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`   | have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image.                                                                                                                                                                                                                               | 200       | 14.46s  | 124       | 6.0       | repetitive; cutoff                   |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `clean-triage-pass` / `clean` | Two tabby cats are lounging on a bright pink couch, each with a remote control nearby. The cat on the left is stretched out on its back, paws in the air, while the one on the right lies curled up, facing away. Both appear relaxed and comfortable in their cozy, colorful setting.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 64        | 6.52s   | 33.7      | 19        | clean                                |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `clean-triage-pass` / `clean` | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 25        | 5.48s   | 22.1      | 15        | clean                                |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `clean-triage-pass` / `clean` | The image shows two cats sleeping on a pink blanket. One cat is a small kitten, and the other is a larger adult cat. Both are lying on their sides, and there are two remote controls placed near them on the couch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 48        | 22.33s  | 34.5      | 18        | clean                                |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `clean-triage-pass` / `clean` | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing the other cat, with a remote control near its head. The cat on the right is lying on its back, also with a remote control nearby. Both cats have striped fur, and the image has a high-contrast, possibly filtered look. The background is a pink couch or cushion. Need to be concise.◁/think▷Two tabby cats are resting on a pink fabric surface. One cat lies on its side facing the other, while the other lies on its back. Both cats have striped fur patterns. A remote control is near each cat. The scene has a high-contrast, stylized visual effect.<\|im_assistant\|>                                                                                                                                                                                                     | 167       | 24.57s  | 79.0      | 15        | thinking-trace                       |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `clean-triage-pass` / `clean` | The image shows two cats lying on a pink blanket. The cat on the left is a kitten with a black and white striped tail. The cat on the right is a tabby cat with a black and brown striped tail. Both cats are sleeping.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 55        | 4.19s   | 63.3      | 9.7       | clean                                |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `clean-triage-pass` / `clean` | Two tabby cats are sleeping peacefully on a bright pink couch. One cat is stretched out on its side, while the other is curled up nearby. Two remote controls lie on the couch between them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 41        | 8.03s   | 19.6      | 30        | clean                                |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `clean-triage-pass` / `clean` | Two tabby cats are peacefully napping on a bright pink couch. The cat on the left is lying on its back with its paws curled up, its tail curled around its body. It has a green collar. The cat on the right is also lying on its back, with its head resting on its paws and its tail stretched out. Between the two cats are two television remotes, one white and one blue. The couch is made of a soft, plush material. The lighting in the room is soft and warm, creating a cozy atmosphere. The cats appear relaxed and content, enjoying a comfortable nap.                                                                                                                                                                                                                                                                                                                                                                   | 124       | 6.09s   | 48.6      | 17        | clean                                |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `clean-triage-pass` / `clean` | Two cats are sleeping on a pink blanket on a red couch, with two remote controls nearby.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 20        | 14.75s  | 33.1      | 19        | clean                                |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`   | ◁think▷Okay, let me try to figure out how to describe this image. First, I need to look at the image carefully. There are two cats on a pink surface, maybe a blanket or a bed. Both cats seem to be lying down. The one on the left is a tabby with gray and black stripes, and the one on the right is a darker tabby, maybe a bit bigger. They both look like they're sleeping or resting. There's a remote control near the left cat and another one near the right cat. The background is pink, which might be a couch or a bed. The image is a bit blurry, especially around the edges, which makes it look like a low-resolution or heavily processed photo. The colors are quite saturated, especially the pink background and the cats' fur. The cats' eyes are closed, and their poses are relaxed. Maybe they're siblings or just two cats that are comfortable with each other. The remotes suggest a home setting        | 200       | 25.66s  | 69.1      | 20        | cutoff; thinking-incomplete          |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `clean-triage-pass` / `clean` | Here's a brief description of the image: The image shows two tabby cats lying side-by-side on a bright pink surface, likely a couch cushion or blanket. Both cats are relaxed and appear to be sleeping. Each cat has a remote control resting on its body – one on its chest and one nearby. The cats have striped fur patterns and are in a comfortable, stretched-out position. The overall scene is cute and playful.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 91        | 7.48s   | 31.8      | 18        | clean                                |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `clean-triage-pass` / `clean` | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and is laying on its side with its head facing the right side of the image. The cat on the right is a brown tabby cat with black stripes and is laying on its side with its head facing the left side of the image. There is a gray remote control on the left side of the image and a gray remote control on the right side of the image.                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 101       | 6.00s   | 34.9      | 11        | clean                                |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `clean-triage-pass` / `clean` | The image features two cats lying on a pink surface, possibly a couch or bed. The cats are positioned in a relaxed manner, with one cat lying on its back and the other on its side. Both cats have a tabby pattern in their fur, characterized by stripes and swirls. The background includes a remote control, suggesting a domestic setting. The overall scene conveys a sense of comfort and leisure, typical of a cat's resting behavior.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 92        | 19.95s  | 29.9      | 20        | clean                                |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `clean-triage-pass` / `clean` | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black stripes. Both cats are curled up with their tails tucked in. Two remote controls are visible on the couch behind them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 55        | 6.06s   | 51.5      | 20        | clean                                |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `clean-triage-pass` / `clean` | Here's a brief description of the image: The image shows two tabby kittens lying on their backs on a bright pink surface (likely a couch or blanket). Both kittens are relaxed and appear to be sleeping. Each kitten has a remote control resting on its belly. The kittens have striped fur patterns and are in a playful, endearing pose.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 70        | 12.21s  | 17.9      | 32        | clean                                |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `cutoff degraded`   | Here are my reasoning steps: The user asks: "Describe this image briefly." The image shows two cats sleeping on a pink couch, with a remote control and a DVD case (or something) on the couch. The cats are both tabby, one is curled up, the other is stretched out. The remote is near the left cat. The DVD case is near the right cat. The couch is pink. The cats appear relaxed. The description should be brief. The user wants a brief description. So we can respond with a short paragraph or a few sentences. We need to ensure we comply with policy. There's no disallowed content. It's a simple image description. So we can comply. We should describe the main elements: two tabby cats sleeping on a pink couch, one curled up, the other stretched out, a remote control and a DVD case (or a book) near them. The couch is pink. The cats have striped fur. The remote is white                                  | 200       | 11.05s  | 44.3      | 14        | cutoff; reasoning-leak               |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `clean-triage-pass` / `clean` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray — creating a cozy, domestic scene.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 35        | 15.94s  | 70.0      | 71        | clean                                |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `clean-triage-pass` / `clean` | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black stripes. Both cats are curled up with their tails tucked in. Two remote controls are visible on the couch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 53        | 13.75s  | 30.1      | 27        | clean                                |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `avoid` / `cutoff degraded`   | In the tranquil setting of this image, two feline companions, one adult and one kitten, are enjoying a peaceful slumber on a vibrant red couch. The adult cat, with its coat of brown and black stripes, is curled up on the right side of the couch, embodying the essence of serenity. Its head is gently resting on its paws, a picture of contentment. On the left side of the couch, a kitten with a similar brown and black striped coat is sprawled out in a relaxed pose. Its head is resting on the armrest of the couch, mirroring the adult cat's position. The kitten's body is stretched out, embodying the carefree spirit of youth. Adding a touch of everyday life to this serene scene are two remote controls resting on the armrest of the couch. One is white and the other is blue, their presence suggesting recent human activity. The remote controls, along with the cats, create a harmonious               | 200       | 7.50s   | 40.1      | 15        | cutoff                               |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `clean-triage-pass` / `clean` | A top-down view of two cats sleeping on a pink blanket. The cat on the left is a gray and black tabby cat, and it is lying on its side with its head facing the right and its front paws stretched out in front of it. Its tail is sticking out to the left. The cat on the right is lying on its side, and it is brown and black. Its head is facing the left, and its front paws are stretched out in front of it. Its tail is sticking out to the left. There are two white and blue remote controls between the cats.                                                                                                                                                                                                                                                                                                                                                                                                             | 119       | 8.94s   | 19.6      | 10        | clean                                |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `clean-triage-pass` / `clean` | Two tabby cats are sleeping peacefully on a bright pink couch, each nestled beside a remote control — one white, one with colorful buttons — creating a cozy and slightly humorous scene of feline comfort.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 41        | 20.64s  | 60.6      | 71        | clean                                |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `clean-triage-pass` / `clean` | The image shows two tabby cats resting on a bright pink blanket spread over a red couch. The smaller cat (left) lies stretched out with a green collar, while the larger cat (right) curls slightly. Two remote controls—one gray, one white—rest near the cats, adding a cozy, domestic feel to the scene. <\|begin_of_box\|>Two tabby cats rest on a pink blanket over a red couch, with two remote controls nearby.<\|end_of_box\|>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 91        | 18.03s  | 49.9      | 63        | clean                                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`   | Okay, let's break this down. The user wants a brief description of an image, and they've given me a real head-scratcher of a prompt. "Red cat, red sofa, remote control, tabby cat, sleeping, on the red sofa, with a remote control, and a tabby cat, sleeping on the red sofa." Hmm. First, I need to figure out what the user *actually* wants. They've given me a jumbled mess of words, and I need to extract the core meaning. It's like they're trying to describe a picture, but the words are all mixed up. I see "red cat," "red sofa," "remote control," "tabby cat," "sleeping," and "on the red sofa." Okay, I get it. They want a description of a picture with a red cat, a red sofa, a remote control, and a tabby cat, both sleeping on                                                                                                                                                                              | 200       | 32.37s  | 61.6      | 60        | cutoff; degeneration                 |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `clean-triage-pass` / `clean` | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 36        | 19.04s  | 7.58      | 63        | clean                                |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `cutoff degraded`   | In the tranquil setting of this image, two feline companions, one a tabby cat and the other a Bengal cat, are enjoying a peaceful slumber on a vibrant red couch. The tabby cat, with its distinctive striped coat, is sprawled out on its side, its head resting comfortably on the armrest of the couch. Its body is relaxed, embodying the essence of serenity. Beside it, the Bengal cat is curled up in a perfect circle, its head resting on its paws in a picture of contentment. The contrast between the two cats' sleeping positions adds a touch of whimsy to the scene. Adding to the homely atmosphere, two remote controls are casually placed on the couch. One is near the tabby cat, while the other is closer to the Bengal cat. Their presence suggests recent human activity, adding a layer of lived-in warmth to the scene. The image captures a moment of calm and companionship between two cats, set against | 200       | 14.01s  | 19.7      | 27        | cutoff                               |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `clean-triage-pass` / `clean` | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a kitten with a fluffy tail and a green collar, while the cat on the right is an adult with a striped pattern. The image appears to be a playful and cozy scene, with the cats seemingly enjoying a relaxing moment together.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 77        | 24.00s  | 5.08      | 25        | clean                                |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `clean-triage-pass` / `clean` | This image captures a cozy scene of two cats lounging on a pink blanket. The cats are positioned side by side, with one on the left and one on the right. Both felines have similar fur patterns, featuring a mix of brown, black, and white stripes. They appear to be in a relaxed state, with their eyes closed and bodies stretched out comfortably. The cat on the left has its head resting on a white remote control, while the cat on the right is nestled next to another white remote. The pink blanket they're lying on is slightly wrinkled, adding to the casual and homey atmosphere of the scene. The background is a solid pink color, which complements the blanket and creates a warm, inviting setting. This image beautifully captures a moment of feline tranquility and comfort.                                                                                                                                | 163       | 47.86s  | 5.65      | 23        | clean                                |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `clean-triage-pass` / `clean` | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and a black tail. It is laying on its side with its head on the pink blanket and its body facing the left side of the image. Its front legs are stretched out in front of it, and its back legs are curled up. The cat on the right is a brown tabby cat with black stripes. It is laying on its side with its head on the pink blanket and its body facing the right side of the image. Its front legs are stretched out in front of it, and its back legs are curled up. There is a gray remote control on the left side of the image and a gray remote control on the right side of the image.                                                                                                                                                                                                                    | 159       | 33.49s  | 5.37      | 26        | clean                                |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`   | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing the other cat, with a remote control near its head. The cat on the right is lying on its back, also with a remote control near it. Both cats have striped fur with a mix of brown, black, and white. The image has a high-contrast, almost posterized or filtered look, making colors very vivid. The background is a pinkish-red fabric, maybe a couch or bed. The remotes are light blue with buttons, typical of TV remotes. The overall style is stylized with high saturation and contrast.◁/think▷The image shows two tabby cats resting on a pink fabric surface. The cat on the left lies on its side, facing the other cat, with a light - colored remote control near its head. The cat on the right is positioned                                                          | 200       | 69.24s  | 4.53      | 39        | cutoff; thinking-trace               |
<!-- markdownlint-enable MD034 MD049 -->

## Quick Navigation

- _Successful outputs:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava),
  [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit),
  +47 more
- _Flagged outputs:_ [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit),
  [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8),
  [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16),
  [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx),
  [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit),
  [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit),
  +11 more
- _Failed outputs:_ [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit),
  [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8),
  [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16),
  [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ❌ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `mlx-vlm`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect prompt-template, stop-token, and decode post-processing
  behavior.
- _Error summary:_ Model loading failed: Server disconnected without sending a
  response.
- _Key signals:_ model error; mlx vlm model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_VLM_MODEL_LOAD_MODEL`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
    yield
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
        pool_request.request
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 793, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 444, in get_model_path
    snapshot_download(
    ~~~~~~~~~~~~~~~~~^
        repo_id=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<11 lines>...
        force_download=force_download,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 259, in snapshot_download
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3630, in repo_info
    return method(
        repo_id,
    ...<4 lines>...
        files_metadata=files_metadata,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3309, in model_info
    r = get_session().get(path, headers=headers, timeout=timeout, params=params)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1053, in get
    return self.request(
           ~~~~~~~~~~~~^
        "GET",
        ^^^^^^
    ...<7 lines>...
        extensions=extensions,
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
        request,
    ...<2 lines>...
        history=[],
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
        request,
        follow_redirects=follow_redirects,
        history=history,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 249, in handle_request
    with map_httpcore_exceptions():
         ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model loading failed: Server disconnected without sending a response.
```

</details>

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ❌ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `mlx-vlm`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect prompt-template, stop-token, and decode post-processing
  behavior.
- _Error summary:_ Model loading failed: Server disconnected without sending a
  response.
- _Key signals:_ model error; mlx vlm model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_VLM_MODEL_LOAD_MODEL`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
    yield
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
        pool_request.request
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 793, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 444, in get_model_path
    snapshot_download(
    ~~~~~~~~~~~~~~~~~^
        repo_id=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<11 lines>...
        force_download=force_download,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 259, in snapshot_download
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3630, in repo_info
    return method(
        repo_id,
    ...<4 lines>...
        files_metadata=files_metadata,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3309, in model_info
    r = get_session().get(path, headers=headers, timeout=timeout, params=params)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1053, in get
    return self.request(
           ~~~~~~~~~~~~^
        "GET",
        ^^^^^^
    ...<7 lines>...
        extensions=extensions,
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
        request,
    ...<2 lines>...
        history=[],
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
        request,
        follow_redirects=follow_redirects,
        history=history,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 249, in handle_request
    with map_httpcore_exceptions():
         ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model loading failed: Server disconnected without sending a response.
```

</details>

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ❌ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `mlx-vlm`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect prompt-template, stop-token, and decode post-processing
  behavior.
- _Error summary:_ Model loading failed: Server disconnected without sending a
  response.
- _Key signals:_ model error; mlx vlm model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_VLM_MODEL_LOAD_MODEL`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
    yield
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
        pool_request.request
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 793, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 444, in get_model_path
    snapshot_download(
    ~~~~~~~~~~~~~~~~~^
        repo_id=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<11 lines>...
        force_download=force_download,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 259, in snapshot_download
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3630, in repo_info
    return method(
        repo_id,
    ...<4 lines>...
        files_metadata=files_metadata,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3309, in model_info
    r = get_session().get(path, headers=headers, timeout=timeout, params=params)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1053, in get
    return self.request(
           ~~~~~~~~~~~~^
        "GET",
        ^^^^^^
    ...<7 lines>...
        extensions=extensions,
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
        request,
    ...<2 lines>...
        history=[],
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
        request,
        follow_redirects=follow_redirects,
        history=history,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 249, in handle_request
    with map_httpcore_exceptions():
         ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model loading failed: Server disconnected without sending a response.
```

</details>

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ❌ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `mlx-vlm`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect prompt-template, stop-token, and decode post-processing
  behavior.
- _Error summary:_ Model loading failed: Server disconnected without sending a
  response.
- _Key signals:_ model error; mlx vlm model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_VLM_MODEL_LOAD_MODEL`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
    yield
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
        pool_request.request
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 793, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 444, in get_model_path
    snapshot_download(
    ~~~~~~~~~~~~~~~~~^
        repo_id=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<11 lines>...
        force_download=force_download,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 259, in snapshot_download
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3630, in repo_info
    return method(
        repo_id,
    ...<4 lines>...
        files_metadata=files_metadata,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3309, in model_info
    r = get_session().get(path, headers=headers, timeout=timeout, params=params)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1053, in get
    return self.request(
           ~~~~~~~~~~~~^
        "GET",
        ^^^^^^
    ...<7 lines>...
        extensions=extensions,
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
        request,
    ...<2 lines>...
        history=[],
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
        request,
        follow_redirects=follow_redirects,
        history=history,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 249, in handle_request
    with map_httpcore_exceptions():
         ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model loading failed: Server disconnected without sending a response.
```

</details>

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ❌ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `mlx-vlm`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect prompt-template, stop-token, and decode post-processing
  behavior.
- _Error summary:_ Model loading failed: Server disconnected without sending a
  response.
- _Key signals:_ model error; mlx vlm model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_VLM_MODEL_LOAD_MODEL`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
    yield
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
        pool_request.request
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 793, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 444, in get_model_path
    snapshot_download(
    ~~~~~~~~~~~~~~~~~^
        repo_id=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<11 lines>...
        force_download=force_download,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 259, in snapshot_download
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3630, in repo_info
    return method(
        repo_id,
    ...<4 lines>...
        files_metadata=files_metadata,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3309, in model_info
    r = get_session().get(path, headers=headers, timeout=timeout, params=params)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1053, in get
    return self.request(
           ~~~~~~~~~~~~^
        "GET",
        ^^^^^^
    ...<7 lines>...
        extensions=extensions,
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
        request,
    ...<2 lines>...
        history=[],
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
        request,
        follow_redirects=follow_redirects,
        history=history,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 249, in handle_request
    with map_httpcore_exceptions():
         ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model loading failed: Server disconnected without sending a response.
```

</details>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ❌ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `mlx-vlm`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect prompt-template, stop-token, and decode post-processing
  behavior.
- _Error summary:_ Model loading failed: Server disconnected without sending a
  response.
- _Key signals:_ model error; mlx vlm model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_VLM_MODEL_LOAD_MODEL`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
    yield
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
        pool_request.request
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 793, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 444, in get_model_path
    snapshot_download(
    ~~~~~~~~~~~~~~~~~^
        repo_id=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<11 lines>...
        force_download=force_download,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 259, in snapshot_download
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3630, in repo_info
    return method(
        repo_id,
    ...<4 lines>...
        files_metadata=files_metadata,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3309, in model_info
    r = get_session().get(path, headers=headers, timeout=timeout, params=params)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1053, in get
    return self.request(
           ~~~~~~~~~~~~^
        "GET",
        ^^^^^^
    ...<7 lines>...
        extensions=extensions,
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
        request,
    ...<2 lines>...
        history=[],
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
        request,
        follow_redirects=follow_redirects,
        history=history,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 249, in handle_request
    with map_httpcore_exceptions():
         ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.RemoteProtocolError: Server disconnected without sending a response.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model loading failed: Server disconnected without sending a response.
```

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 31.06s; Gen 0.07s; Total 31.14s
- _Throughput:_ Prompt 2,440 TPS (80 tok); Gen 504 TPS (10 tok)
- _Tokens:_ prompt 80 tok; estimated text 6 tok; estimated non-text 74 tok;
  generated 10 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are laying on a pink couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 4.67s; Gen 0.13s; Total 4.80s
- _Throughput:_ Prompt 3,151 TPS (269 tok); Gen 313 TPS (10 tok)
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 10 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.36s; Gen 0.16s; Total 2.53s
- _Throughput:_ Prompt 3,093 TPS (269 tok); Gen 188 TPS (13 tok)
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### ✅ mlx-community/MiniCPM-V-4.6-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.62s; Gen 0.21s; Total 2.84s
- _Throughput:_ Prompt 2,226 TPS (228 tok); Gen 259 TPS (17 tok)
- _Tokens:_ prompt 228 tok; estimated text 6 tok; estimated non-text 222 tok;
  generated 17 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two cats lying on a pink blanket, with remote controls
> nearby.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.93s; Gen 0.31s; Total 1.24s
- _Throughput:_ Prompt 324 TPS (22 tok); Gen 369 TPS (81 tok)
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 81 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows a close-up view of two cats lying down on a pink fabric
> surface. Both cats have striped coats, and they are positioned on a couch.
> The cats are facing the camera, and their eyes are open, indicating
> alertness. The image is a close-up, color photograph, and it captures the
> relaxed posture of the cats. There are no texts or other objects in the
> image.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.68s; Gen 0.33s; Total 2.02s
- _Throughput:_ Prompt 2,876 TPS (417 tok); Gen 325 TPS (52 tok)
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 52 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two cats lying on a pink blanket. One cat is on the left
> side, while the other is on the right side. There are two remote controls
> placed on the blanket, one on the left side and the other on the right side.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.71s; Gen 0.42s; Total 1.14s
- _Throughput:_ Prompt 246 TPS (22 tok); Gen 116 TPS (35 tok)
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 35 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image features two cats lying on a couch. One cat is a light brown and
> the other is a dark brown. They both have green eyes and a black nose.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 5.39s; Gen 0.56s; Total 5.98s
- _Throughput:_ Prompt 739 TPS (286 tok); Gen 128 TPS (21 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 21 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are lying on a pink blanket on a red couch, with two remote
> controls nearby.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 5.27s; Gen 0.58s; Total 5.86s
- _Throughput:_ Prompt 3,051 TPS (770 tok); Gen 60.4 TPS (19 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 19 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch with remote controls beside them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 14.44s; Gen 0.60s; Total 15.04s
- _Throughput:_ Prompt 2,911 TPS (770 tok); Gen 60.7 TPS (19 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 19 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch with remote controls beside them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 53.49s; Gen 0.62s; Total 54.11s
- _Throughput:_ Prompt 2,653 TPS (1,196 tok); Gen 121 TPS (13 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.82s; Gen 0.77s; Total 2.60s
- _Throughput:_ Prompt 3,726 TPS (973 tok); Gen 196 TPS (97 tok)
- _Tokens:_ prompt 973 tok; estimated text 6 tok; estimated non-text 967 tok;
  generated 97 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image depicts two cats lying on a soft, pink cushion or blanket.
>
> &#45; The cat on the left appears to be a kitten, with a smaller size, striped
> tabby pattern, and seems relaxed.
> &#45; The cat on the right is an adult tabby cat, also striped, and is lying on
> its side with its head slightly raised, looking calm and content.
> Both are positioned near what looks like a remote control, suggesting a
> cozy, indoor setting.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Instruct-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.32s; Gen 0.78s; Total 4.10s
- _Throughput:_ Prompt 3,780 TPS (315 tok); Gen 135 TPS (89 tok)
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 89 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This is a close-up photograph of two cats sleeping on a bright pink couch.
> The cat on the left is a tabby with a black and grey pattern, lying on its
> back with its eyes closed. The cat on the right is a calico with a brown and
> black striped pattern, also lying on its back. Both cats are resting
> peacefully. There are two remote controls on the couch, one white and one
> with a blue button.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 46.87s; Gen 0.79s; Total 47.69s
- _Throughput:_ Prompt 3,505 TPS (315 tok); Gen 134 TPS (89 tok)
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 89 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This is a close-up photograph of two cats sleeping on a bright pink couch.
> The cat on the left is a tabby with a black and grey pattern, lying on its
> back with its eyes closed. The cat on the right is a calico with a brown and
> black striped pattern, also lying on its back. Both cats are resting
> peacefully. There are two remote controls on the couch, one white and one
> with a blue button.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 4.65s; Gen 0.81s; Total 5.47s
- _Throughput:_ Prompt 1,290 TPS (408 tok); Gen 89.3 TPS (42 tok)
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 42 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two tabby cats resting on a bright pink couch. One cat lies
> on its side, while the other is curled with its head down. Two remote
> controls are also visible on the couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 7.96s; Gen 0.82s; Total 8.79s
- _Throughput:_ Prompt 857 TPS (319 tok); Gen 117 TPS (51 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 51 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sprawled out asleep on a bright pink couch, nestled
> between two remote controls. The cat on the left is curled with its tail
> tucked, while the one on the right lies stretched out, both appearing deeply
> relaxed and comfortable.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 4.87s; Gen 1.07s; Total 5.95s
- _Throughput:_ Prompt 721 TPS (319 tok); Gen 98.5 TPS (61 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 61 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sprawled out asleep on a bright pink couch, nestled
> between two remote controls — one white and one gray. The cat on the left is
> curled with its tail tucked, while the one on the right lies stretched out,
> both appearing deeply relaxed in a cozy, domestic scene.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a visual input burden issue first; reduce that
  input load or inspect long-context handling before judging output quality.
- _Key signals:_ Output appears truncated to about 3 tokens.; At visual input
  burden (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%;
  weak text signal truncated).; output/prompt=0.07%; visual input burden=100%
- _Timing:_ Load 1.44s; Gen 1.15s; Total 2.60s
- _Throughput:_ Prompt 3,770 TPS (4,103 tok); Gen 82.3 TPS (3 tok)
- _Tokens:_ prompt 4103 tok; estimated text 6 tok; estimated non-text 4097
  tok; generated 3 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Cat.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- output:truncated(3tok)
- long_context_low_ratio(0.1%;4103->3;truncated)

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 25.52s; Gen 1.28s; Total 26.81s
- _Throughput:_ Prompt 1,795 TPS (417 tok); Gen 65.7 TPS (65 tok)
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 65 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats lying on a pink couch. Both cats are
> stretched out, appearing relaxed. Between them are two remote controls. The
> couch has a vibrant pink cover, and the cats' fur has a mix of brown, black,
> and white stripes. The scene conveys a cozy and calm atmosphere.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ formatting=Unknown tags: &lt;channel|&gt;
- _Timing:_ Load 5.10s; Gen 1.35s; Total 6.48s
- _Throughput:_ Prompt 948 TPS (282 tok); Gen 27.9 TPS (25 tok)
- _Tokens:_ prompt 282 tok; estimated text 6 tok; estimated non-text 276 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|channel>thought
> <channel|>A high-angle shot shows two tabby cats sleeping on a pink blanket
> next to two remote controls.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Unknown tags: <channel\|>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-mxfp8

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ formatting=Unknown tags: &lt;channel|&gt;
- _Timing:_ Load 5.06s; Gen 1.38s; Total 6.45s
- _Throughput:_ Prompt 979 TPS (282 tok); Gen 24.9 TPS (23 tok)
- _Tokens:_ prompt 282 tok; estimated text 6 tok; estimated non-text 276 tok;
  generated 23 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|channel>thought
> <channel|>An overhead shot shows two tabby cats sleeping on a pink blanket
> next to two remote controls.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Unknown tags: <channel\|>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; text-sanity=gibberish(token_noise)
- _Timing:_ Load 2.98s; Gen 1.51s; Total 4.51s
- _Throughput:_ Prompt 3,813 TPS (317 tok); Gen 135 TPS (188 tok)
- _Tokens:_ prompt 317 tok; estimated text 6 tok; estimated non-text 311 tok;
  generated 188 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> So, let's see. The image shows two cats lying on a pink couch. The couch is
> a bright pink color, and there are two remote controls next to the cats. One
> cat is on the left, with a striped pattern, and the other is on the right,
> also striped but maybe a bit different. Both cats are relaxed, lying down,
> maybe sleeping. The remotes are white with some buttons, and one has a blue
> button. The background is the pink couch, so the main elements are the two
> cats, the remotes, and the pink fabric. I need to describe this briefly.
> &lt;/think&gt;
>
> Two tabby cats are lying on a bright pink couch, relaxed and asleep. A white
> remote control with a blue button sits between them, and another remote
> control is positioned near the left cat. The scene is cozy, with the cats'
> striped fur and the vibrant pink fabric of the couch forming the backdrop.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (gibberish(token_noise))
- token_leak:&lt;/think&gt;

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.61s; Gen 1.53s; Total 3.15s
- _Throughput:_ Prompt 1,440 TPS (974 tok); Gen 64.1 TPS (54 tok)
- _Tokens:_ prompt 974 tok; estimated text 6 tok; estimated non-text 968 tok;
  generated 54 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two tabby cats lounging on a pink cushion. Both cats appear
> relaxed, lying on their sides and stretching out their paws. One cat faces
> left, while the other faces right, and there's a remote control near each of
> them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.86s; Gen 1.57s; Total 3.44s
- _Throughput:_ Prompt 1,463 TPS (974 tok); Gen 67.5 TPS (60 tok)
- _Tokens:_ prompt 974 tok; estimated text 6 tok; estimated non-text 968 tok;
  generated 60 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two cats lying on their backs on a pink blanket or cushion.
> Both cats appear relaxed and are positioned close to each other, with their
> paws in the air. There are two remote controls placed near them, suggesting
> a cozy indoor setting, possibly on a couch or bed.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.69s; Gen 1.61s; Total 5.31s
- _Throughput:_ Prompt 395 TPS (286 tok); Gen 28.7 TPS (25 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping on a bright pink blanket on a red couch, with
> two remote controls lying next to them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ formatting=Unknown tags: &lt;end_of_utterance&gt;
- _Timing:_ Load 31.33s; Gen 1.73s; Total 33.07s
- _Throughput:_ Prompt 2,301 TPS (2,327 tok); Gen 34.2 TPS (23 tok)
- _Tokens:_ prompt 2327 tok; estimated text 6 tok; estimated non-text 2321
  tok; generated 23 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In this image we can see two cats on the sofa. There are two remotes on the
> sofa.<end_of_utterance>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Unknown tags: <end_of_utterance>

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); repetitive token=phrase: "have this
  image. have..."
- _Timing:_ Load 12.59s; Gen 1.86s; Total 14.46s
- _Throughput:_ Prompt 1,152 TPS (266 tok); Gen 124 TPS (200 tok)
- _Tokens:_ prompt 266 tok; estimated text 6 tok; estimated non-text 260 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "have this image. have...")
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.98s; Gen 2.52s; Total 6.52s
- _Throughput:_ Prompt 530 TPS (319 tok); Gen 33.7 TPS (64 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 64 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are lounging on a bright pink couch, each with a remote
> control nearby. The cat on the left is stretched out on its back, paws in
> the air, while the one on the right lies curled up, facing away. Both appear
> relaxed and comfortable in their cozy, colorful setting.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.92s; Gen 2.55s; Total 5.48s
- _Throughput:_ Prompt 11.5 TPS (16 tok); Gen 22.1 TPS (25 tok)
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats lying on a pink blanket, with two remote
> controls placed on the couch behind them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 19.66s; Gen 2.66s; Total 22.33s
- _Throughput:_ Prompt 2,693 TPS (3,341 tok); Gen 34.5 TPS (48 tok)
- _Tokens:_ prompt 3341 tok; estimated text 6 tok; estimated non-text 3335
  tok; generated 48 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two cats sleeping on a pink blanket. One cat is a small
> kitten, and the other is a larger adult cat. Both are lying on their sides,
> and there are two remote controls placed near them on the couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ thinking trace present
- _Timing:_ Load 21.81s; Gen 2.76s; Total 24.57s
- _Throughput:_ Prompt 921 TPS (399 tok); Gen 79.0 TPS (167 tok)
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 167 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's describe the image. First, there are two tabby cats on a
> pink fabric surface. The cat on the left is lying on its side, facing the
> other cat, with a remote control near its head. The cat on the right is
> lying on its back, also with a remote control nearby. Both cats have striped
> fur, and the image has a high-contrast, possibly filtered look. The
> background is a pink couch or cushion. Need to be concise.◁/think▷Two tabby
> cats are resting on a pink fabric surface. One cat lies on its side facing
> the other, while the other lies on its back. Both cats have striped fur
> patterns. A remote control is near each cat. The scene has a high-contrast,
> stylized visual effect.<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Thinking trace present (expected model protocol; ◁think▷)

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.39s; Gen 2.79s; Total 4.19s
- _Throughput:_ Prompt 1,241 TPS (2,356 tok); Gen 63.3 TPS (55 tok)
- _Tokens:_ prompt 2356 tok; estimated text 6 tok; estimated non-text 2350
  tok; generated 55 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two cats lying on a pink blanket. The cat on the left is a
> kitten with a black and white striped tail. The cat on the right is a tabby
> cat with a black and brown striped tail. Both cats are sleeping.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 5.19s; Gen 2.82s; Total 8.03s
- _Throughput:_ Prompt 451 TPS (319 tok); Gen 19.6 TPS (41 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 41 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping peacefully on a bright pink couch. One cat is
> stretched out on its side, while the other is curled up nearby. Two remote
> controls lie on the couch between them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.19s; Gen 2.89s; Total 6.09s
- _Throughput:_ Prompt 850 TPS (274 tok); Gen 48.6 TPS (124 tok)
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 124 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are peacefully napping on a bright pink couch. The cat on the
> left is lying on its back with its paws curled up, its tail curled around
> its body. It has a green collar. The cat on the right is also lying on its
> back, with its head resting on its paws and its tail stretched out.
>
> Between the two cats are two television remotes, one white and one blue. The
> couch is made of a soft, plush material. The lighting in the room is soft
> and warm, creating a cozy atmosphere. The cats appear relaxed and content,
> enjoying a comfortable nap.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 11.77s; Gen 2.97s; Total 14.75s
- _Throughput:_ Prompt 1,486 TPS (3,341 tok); Gen 33.1 TPS (20 tok)
- _Tokens:_ prompt 3341 tok; estimated text 6 tok; estimated non-text 3335
  tok; generated 20 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a red couch, with two remote
> controls nearby.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); thinking trace incomplete
- _Timing:_ Load 22.32s; Gen 3.33s; Total 25.66s
- _Throughput:_ Prompt 1,319 TPS (399 tok); Gen 69.1 TPS (200 tok)
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Okay, let me try to figure out how to describe this image. First, I
> need to look at the image carefully. There are two cats on a pink surface,
> maybe a blanket or a bed. Both cats seem to be lying down. The one on the
> left is a tabby with gray and black stripes, and the one on the right is a
> darker tabby, maybe a bit bigger. They both look like they're sleeping or
> resting. There's a remote control near the left cat and another one near the
> right cat. The background is pink, which might be a couch or a bed. The
> image is a bit blurry, especially around the edges, which makes it look like
> a low-resolution or heavily processed photo. The colors are quite
> saturated, especially the pink background and the cats' fur. The cats' eyes
> are closed, and their poses are relaxed. Maybe they're siblings or just two
> cats that are comfortable with each other. The remotes suggest a home
> setting
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Thinking trace incomplete (expected model protocol; opened with ◁think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.90s; Gen 3.57s; Total 7.48s
- _Throughput:_ Prompt 399 TPS (275 tok); Gen 31.8 TPS (91 tok)
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 91 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here's a brief description of the image:
>
> The image shows two tabby cats lying side-by-side on a bright pink surface,
> likely a couch cushion or blanket. Both cats are relaxed and appear to be
> sleeping. Each cat has a remote control resting on its body – one on its
> chest and one nearby. The cats have striped fur patterns and are in a
> comfortable, stretched-out position. The overall scene is cute and playful.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.24s; Gen 3.76s; Total 6.00s
- _Throughput:_ Prompt 1,327 TPS (1,031 tok); Gen 34.9 TPS (101 tok)
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 101 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A top-down view of two cats laying on a pink blanket. The cat on the left is
> a gray tabby cat with black stripes and is laying on its side with its head
> facing the right side of the image. The cat on the right is a brown tabby
> cat with black stripes and is laying on its side with its head facing the
> left side of the image. There is a gray remote control on the left side of
> the image and a gray remote control on the right side of the image.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 16.13s; Gen 3.81s; Total 19.95s
- _Throughput:_ Prompt 617 TPS (441 tok); Gen 29.9 TPS (92 tok)
- _Tokens:_ prompt 441 tok; estimated text 6 tok; estimated non-text 435 tok;
  generated 92 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image features two cats lying on a pink surface, possibly a couch or
> bed. The cats are positioned in a relaxed manner, with one cat lying on its
> back and the other on its side. Both cats have a tabby pattern in their fur,
> characterized by stripes and swirls. The background includes a remote
> control, suggesting a domestic setting. The overall scene conveys a sense of
> comfort and leisure, typical of a cat's resting behavior.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.45s; Gen 4.59s; Total 6.06s
- _Throughput:_ Prompt 213 TPS (745 tok); Gen 51.5 TPS (55 tok)
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 55 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a red couch. One cat is thin
> with black stripes, while the other is thicker with brown and black stripes.
> Both cats are curled up with their tails tucked in. Two remote controls are
> visible on the couch behind them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 7.43s; Gen 4.77s; Total 12.21s
- _Throughput:_ Prompt 331 TPS (275 tok); Gen 17.9 TPS (70 tok)
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 70 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here's a brief description of the image:
>
> The image shows two tabby kittens lying on their backs on a bright pink
> surface (likely a couch or blanket). Both kittens are relaxed and appear to
> be sleeping. Each kitten has a remote control resting on its belly. The
> kittens have striped fur patterns and are in a playful, endearing pose.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); reasoning leak
- _Timing:_ Load 5.58s; Gen 5.45s; Total 11.05s
- _Throughput:_ Prompt 1,434 TPS (1,330 tok); Gen 44.3 TPS (200 tok)
- _Tokens:_ prompt 1330 tok; estimated text 6 tok; estimated non-text 1324
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here are my reasoning steps:
> The user asks: "Describe this image briefly." The image shows two cats
> sleeping on a pink couch, with a remote control and a DVD case (or
> something) on the couch. The cats are both tabby, one is curled up, the
> other is stretched out. The remote is near the left cat. The DVD case is
> near the right cat. The couch is pink. The cats appear relaxed. The
> description should be brief. The user wants a brief description. So we can
> respond with a short paragraph or a few sentences.
>
> We need to ensure we comply with policy. There's no disallowed content. It's
> a simple image description. So we can comply.
>
> We should describe the main elements: two tabby cats sleeping on a pink
> couch, one curled up, the other stretched out, a remote control and a DVD
> case (or a book) near them. The couch is pink. The cats have striped fur.
> The remote is white
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (here are my reasoning steps, the user asks:)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 10.47s; Gen 5.46s; Total 15.94s
- _Throughput:_ Prompt 64.5 TPS (319 tok); Gen 70.0 TPS (35 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 35 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sprawled out asleep on a bright pink couch, nestled
> between two remote controls — one white and one gray — creating a cozy,
> domestic scene.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 8.10s; Gen 5.64s; Total 13.75s
- _Throughput:_ Prompt 193 TPS (745 tok); Gen 30.1 TPS (53 tok)
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 53 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a red couch. One cat is thin
> with black stripes, while the other is thicker with brown and black stripes.
> Both cats are curled up with their tails tucked in. Two remote controls are
> visible on the couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200)
- _Timing:_ Load 1.72s; Gen 5.78s; Total 7.50s
- _Throughput:_ Prompt 1,604 TPS (1,239 tok); Gen 40.1 TPS (200 tok)
- _Tokens:_ prompt 1239 tok; estimated text 6 tok; estimated non-text 1233
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In the tranquil setting of this image, two feline companions, one adult and
> one kitten, are enjoying a peaceful slumber on a vibrant red couch. The
> adult cat, with its coat of brown and black stripes, is curled up on the
> right side of the couch, embodying the essence of serenity. Its head is
> gently resting on its paws, a picture of contentment.
>
> On the left side of the couch, a kitten with a similar brown and black
> striped coat is sprawled out in a relaxed pose. Its head is resting on the
> armrest of the couch, mirroring the adult cat's position. The kitten's body
> is stretched out, embodying the carefree spirit of youth.
>
> Adding a touch of everyday life to this serene scene are two remote controls
> resting on the armrest of the couch. One is white and the other is blue,
> their presence suggesting recent human activity. The remote controls, along
> with the cats, create a harmonious
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.46s; Gen 6.47s; Total 8.94s
- _Throughput:_ Prompt 2,689 TPS (1,031 tok); Gen 19.6 TPS (119 tok)
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 119 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A top-down view of two cats sleeping on a pink blanket. The cat on the left
> is a gray and black tabby cat, and it is lying on its side with its head
> facing the right and its front paws stretched out in front of it. Its tail
> is sticking out to the left. The cat on the right is lying on its side, and
> it is brown and black. Its head is facing the left, and its front paws are
> stretched out in front of it. Its tail is sticking out to the left. There
> are two white and blue remote controls between the cats.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### ✅ mlx-community/Ornith-1.0-35B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 13.61s; Gen 7.01s; Total 20.64s
- _Throughput:_ Prompt 50.5 TPS (319 tok); Gen 60.6 TPS (41 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 41 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping peacefully on a bright pink couch, each nestled
> beside a remote control — one white, one with colorful buttons — creating a
> cozy and slightly humorous scene of feline comfort.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 10.41s; Gen 7.61s; Total 18.03s
- _Throughput:_ Prompt 70.7 TPS (408 tok); Gen 49.9 TPS (91 tok)
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 91 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats resting on a bright pink blanket spread over
> a red couch. The smaller cat (left) lies stretched out with a green collar,
> while the larger cat (right) curls slightly. Two remote controls—one gray,
> one white—rest near the cats, adding a cozy, domestic feel to the scene.
> <|begin_of_box|>Two tabby cats rest on a pink blanket over a red couch, with
> two remote controls nearby.<|end_of_box|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); degeneration=incomplete_sentence: ends
  with 'on'
- _Timing:_ Load 24.23s; Gen 8.13s; Total 32.37s
- _Throughput:_ Prompt 90.3 TPS (439 tok); Gen 61.6 TPS (200 tok)
- _Tokens:_ prompt 439 tok; estimated text 6 tok; estimated non-text 433 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Okay, let's break this down. The user wants a brief description of an image,
> and they've given me a real head-scratcher of a prompt. "Red cat, red sofa,
> remote control, tabby cat, sleeping, on the red sofa, with a remote
> control, and a tabby cat, sleeping on the red sofa." Hmm.
>
> First, I need to figure out what the user &#42;actually&#42; wants. They've given me
> a jumbled mess of words, and I need to extract the core meaning. It's like
> they're trying to describe a picture, but the words are all mixed up. I see
> "red cat," "red sofa," "remote control," "tabby cat," "sleeping," and "on
> the red sofa." Okay, I get it. They want a description of a picture with a
> red cat, a red sofa, a remote control, and a tabby cat, both sleeping on
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'on')
- Generation loop (degeneration)
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 9.18s; Gen 9.85s; Total 19.04s
- _Throughput:_ Prompt 53.9 TPS (274 tok); Gen 7.58 TPS (36 tok)
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 36 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Both cats are sleeping on a pink blanket. The difference between these
> images is that one cat is on the left side of the blanket and the other is
> on the right side.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200)
- _Timing:_ Load 3.03s; Gen 10.97s; Total 14.01s
- _Throughput:_ Prompt 1,554 TPS (1,239 tok); Gen 19.7 TPS (200 tok)
- _Tokens:_ prompt 1239 tok; estimated text 6 tok; estimated non-text 1233
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In the tranquil setting of this image, two feline companions, one a tabby
> cat and the other a Bengal cat, are enjoying a peaceful slumber on a vibrant
> red couch. The tabby cat, with its distinctive striped coat, is sprawled
> out on its side, its head resting comfortably on the armrest of the couch.
> Its body is relaxed, embodying the essence of serenity.
>
> Beside it, the Bengal cat is curled up in a perfect circle, its head resting
> on its paws in a picture of contentment. The contrast between the two cats'
> sleeping positions adds a touch of whimsy to the scene.
>
> Adding to the homely atmosphere, two remote controls are casually placed on
> the couch. One is near the tabby cat, while the other is closer to the
> Bengal cat. Their presence suggests recent human activity, adding a layer of
> lived-in warmth to the scene.
>
> The image captures a moment of calm and companionship between two cats, set
> against
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 6.26s; Gen 17.73s; Total 24.00s
- _Throughput:_ Prompt 6.69 TPS (17 tok); Gen 5.08 TPS (77 tok)
- _Tokens:_ prompt 17 tok; estimated text 6 tok; estimated non-text 11 tok;
  generated 77 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats lying on a pink blanket, with two remote
> controls placed on the couch behind them. The cat on the left is a kitten
> with a fluffy tail and a green collar, while the cat on the right is an
> adult with a striped pattern. The image appears to be a playful and cozy
> scene, with the cats seemingly enjoying a relaxing moment together.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ✅ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 17.96s; Gen 29.88s; Total 47.86s
- _Throughput:_ Prompt 879 TPS (766 tok); Gen 5.65 TPS (163 tok)
- _Tokens:_ prompt 766 tok; estimated text 6 tok; estimated non-text 760 tok;
  generated 163 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image captures a cozy scene of two cats lounging on a pink blanket. The
> cats are positioned side by side, with one on the left and one on the
> right. Both felines have similar fur patterns, featuring a mix of brown,
> black, and white stripes. They appear to be in a relaxed state, with their
> eyes closed and bodies stretched out comfortably.
>
> The cat on the left has its head resting on a white remote control, while
> the cat on the right is nestled next to another white remote. The pink
> blanket they're lying on is slightly wrinkled, adding to the casual and
> homey atmosphere of the scene.
>
> The background is a solid pink color, which complements the blanket and
> creates a warm, inviting setting. This image beautifully captures a moment
> of feline tranquility and comfort.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.69s; Gen 30.79s; Total 33.49s
- _Throughput:_ Prompt 913 TPS (1,031 tok); Gen 5.37 TPS (159 tok)
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 159 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A top-down view of two cats laying on a pink blanket. The cat on the left is
> a gray tabby cat with black stripes and a black tail. It is laying on its
> side with its head on the pink blanket and its body facing the left side of
> the image. Its front legs are stretched out in front of it, and its back
> legs are curled up. The cat on the right is a brown tabby cat with black
> stripes. It is laying on its side with its head on the pink blanket and its
> body facing the right side of the image. Its front legs are stretched out in
> front of it, and its back legs are curled up. There is a gray remote
> control on the left side of the image and a gray remote control on the right
> side of the image.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); thinking trace present
- _Timing:_ Load 23.96s; Gen 45.27s; Total 69.24s
- _Throughput:_ Prompt 454 TPS (399 tok); Gen 4.53 TPS (200 tok)
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's describe the image. First, there are two tabby cats on a
> pink fabric surface. The cat on the left is lying on its side, facing the
> other cat, with a remote control near its head. The cat on the right is
> lying on its back, also with a remote control near it. Both cats have
> striped fur with a mix of brown, black, and white. The image has a
> high-contrast, almost posterized or filtered look, making colors very vivid.
> The background is a pinkish-red fabric, maybe a couch or bed. The remotes
> are light blue with buttons, typical of TV remotes. The overall style is
> stylized with high saturation and contrast.◁/think▷The image shows two tabby
> cats resting on a pink fabric surface. The cat on the left lies on its
> side, facing the other cat, with a light - colored remote control near its
> head. The cat on the right is positioned
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Thinking trace present (expected model protocol; ◁think▷)
- Likely capped by max token budget

---

<!-- markdownlint-enable MD033 MD034 -->
