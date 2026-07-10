# Model Output Gallery

Generated on: 2026-07-10 21:31:49 BST

Complete per-model evidence artifact with image metadata, the source prompt,
summary tables, diagnostics, and full generated output for every attempted
model.

_Action Snapshot: see [results.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.md) for the full summary._

## Run Stamps

- `mlx-vlm`: `0.6.5`
- `mlx`: `0.32.1.dev20260710+4367c73b`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.13.0`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.23.0`
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

## Model Quality Summary

Skim-first view of what each model returned, or the strongest diagnostic when it did not produce usable output.

<!-- markdownlint-disable MD034 -->

| Model                                                                                                                   | Result                            | Quality / diagnostic                 | Response / diagnostic                                                                                                                                                                                                                                                                                                              |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `avoid` / `semantic mismatch`     | mlx-vlm; model-error                 | [mlx-vlm; model-error] Error: Model Error - Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).                                                                                         |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `clean-triage-pass` / `clean`     | clean                                | Two cats are laying on a pink couch.                                                                                                                                                                                                                                                                                               |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket.                                                                                                                                                                                                                                                                                           |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink couch next to two remote controls.                                                                                                                                                                                                                                                                 |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `clean-triage-pass` / `clean`     | clean                                | The image shows two cats lying on a pink blanket, with remote controls nearby.                                                                                                                                                                                                                                                     |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `clean-triage-pass` / `clean`     | clean                                | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats have striped coats, and they are positioned on a couch. The cats are facing the camera, and their eyes are open, indi ... [tail] , and it captures the relaxed posture of the cats. There are no texts or other objects in the image.   |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `clean-triage-pass` / `clean`     | clean                                | The image shows two cats lying on a pink blanket. One cat is on the left side, while the other is on the right side. There are two remote controls placed on the blanket, one on the left side and the other on the right side.                                                                                                    |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `clean-triage-pass` / `clean`     | clean                                | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                                                                                                                                                                 |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                                                                            |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                                                                            |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `clean-triage-pass` / `clean`     | clean                                | This image depicts two cats lying on a soft, pink cushion or blanket. - The cat on the left appears to be a kitten, with a smaller size, striped tabby pattern, and seems relaxed. - The cat on the right is an ... [tail] ntent. Both are positioned near what looks like a remote control, suggesting a cozy, indoor setting.    |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are lying on a pink blanket on a red couch, with two remote controls nearby.                                                                                                                                                                                                                                        |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `clean-triage-pass` / `clean`     | clean                                | This is a close-up photograph of two cats sleeping on a bright pink couch. The cat on the left is a tabby with a black and grey pattern, lying on its back with its eyes closed. The cat on the right is a calic ... [tail] esting peacefully. There are two remote controls on the couch, one white and one with a blue button.   |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `clean-triage-pass` / `clean`     | clean                                | This is a close-up photograph of two cats sleeping on a bright pink couch. The cat on the left is a tabby with a black and grey pattern, lying on its back with its eyes closed. The cat on the right is a calic ... [tail] esting peacefully. There are two remote controls on the couch, one white and one with a blue button.   |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls. The cat on the left is curled with its tail tucked, while the one on the right lies stretched out, both appearing deeply relaxed and comfortable.                                                                              |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `clean-triage-pass` / `clean`     | clean                                | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled with its head down. Two remote controls are also visible on the couch.                                                                                                                                         |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray. The cat on the left is curled with its tail tucked, while the one on the right lies stretched out, both appearing deeply relaxed in a cozy, domestic scene.                                           |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `clean-triage-pass` / `clean`     | clean                                | This image shows two tabby cats sleeping on a bright pink couch. The cat on the left is a smaller kitten, lying on its side with its head resting near a white remote control. The cat on the right is larger an ... [tail] e control. Both cats appear relaxed and comfortable, curled up on the soft pink fabric of the couch.   |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `caveat` / `context budget`       | harness:long-context; context-budget | Cat.                                                                                                                                                                                                                                                                                                                               |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are sleeping on a pink blanket on a red couch next to remote controls.                                                                                                                                                                                                                                              |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `clean-triage-pass` / `clean`     | clean                                | A high-angle shot shows two tabby cats sleeping on a pink blanket next to two remote controls.                                                                                                                                                                                                                                     |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `clean-triage-pass` / `clean`     | clean                                | The image shows two tabby cats lying on a pink couch. Both cats are stretched out, appearing relaxed. Between them are two remote controls. The couch has a vibrant pink cover, and the cats' fur has a mix of brown, black, and white stripes. The scene conveys a cozy and calm atmosphere.                                      |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `clean-triage-pass` / `clean`     | clean                                | The image shows two tabby cats resting on a bright pink couch. One cat lies on its side, eyes closed, with a remote control nearby. The other cat, larger and more robust, is curled slightly, also appearing relaxed. The scene conveys a cozy, peaceful atmosphere.                                                              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `clean-triage-pass` / `clean`     | clean                                | This image shows two tabby cats lounging on a pink cushion. Both cats appear relaxed, lying on their sides and stretching out their paws. One cat faces left, while the other faces right, and there's a remote control near each of them.                                                                                         |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `clean-triage-pass` / `clean`     | clean                                | This image shows two cats lying on their backs on a pink blanket or cushion. Both cats appear relaxed and are positioned close to each other, with their paws in the air. There are two remote controls placed near them, suggesting a cozy indoor setting, possibly on a couch or bed.                                            |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `harness`               | harness:stop-token                   | So, let's see. The image shows two cats lying on a pink couch. The couch is a bright pink color, and there are two remote controls next to the cats. One cat is on the left, with a striped pattern, and the oth ... [tail] e is cozy, with the cats' striped fur and the vibrant pink fabric of the couch forming the backdrop.   |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `clean-triage-pass` / `clean`     | formatting                           | In this image we can see two cats on the sofa. There are two remotes on the sofa.<end_of_utterance>                                                                                                                                                                                                                                |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `clean-triage-pass` / `token cap` | hit token cap (200)                  | In the image, a cat is comfortably nestled on a vibrant red couch. The feline, with its fur a mix of black and white, is facing the camera, its eyes half-closed in what appears to be a state of relaxation. Th ... [tail] e cat seemingly undisturbed by the presence of the remote control. It's a snapshot of everyday life,   |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`       | repetitive; cutoff                   | have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this i...   |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                                                                                                                                                                  |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `clean-triage-pass` / `clean`     | clean                                | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                                                                                                                                                                  |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are lounging on a bright pink couch, each with a remote control nearby. The cat on the left is stretched out on its back, paws in the air, while the one on the right lies curled up, facing away. Both appear relaxed and comfortable in their cozy, colorful setting.                                             |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `clean-triage-pass` / `clean`     | reasoning-leak                       | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing the other cat, with a remote control near its head. The cat on t ... [tail] ote control is near each cat. The scene has a high-contrast, stylized visual effect.<\|im_assistant\|> |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are sleeping peacefully on a bright pink couch, each nestled beside a remote control — one white, one with colorful buttons — creating a cozy and slightly humorous scene of feline comfort.                                                                                                                        |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `clean-triage-pass` / `clean`     | clean                                | The image shows two cats sleeping on a pink blanket. One cat is a small kitten, and the other is a larger adult cat. Both are lying on their sides, and there are two remote controls placed near them on the couch.                                                                                                               |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `clean-triage-pass` / `clean`     | clean                                | The image shows two cats lying on a pink blanket. The cat on the left is a kitten with a black and white striped tail. The cat on the right is a tabby cat with a black and brown striped tail. Both cats are sleeping.                                                                                                            |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray — creating a cozy, domestic scene.                                                                                                                                                                     |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are sleeping peacefully on a bright pink couch. One cat is stretched out on its side, while the other is curled up nearby. Two remote controls lie on the couch between them.                                                                                                                                       |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are peacefully napping on a bright pink couch. The cat on the left is lying on its back with its paws curled up, its tail curled around its body. It has a green collar. The cat on the right is ... [tail] d warm, creating a cozy atmosphere. The cats appear relaxed and content, enjoying a comfortable nap.    |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`       | cutoff; reasoning-leak               | ◁think▷Okay, let me try to figure out how to describe this image. First, I need to look at the image carefully. There are two cats on a pink surface, maybe a blanket or a bed. Both cats seem to be lying down. ... [tail] e siblings or just two cats that are comfortable with each other. The remotes suggest a home setting   |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are sleeping peacefully on a bright pink couch. One cat lies on its side near a white remote control, while the other stretches out nearby, also next to a remote. Both appear relaxed and comfortable in their cozy spot.                                                                                          |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a red couch, with two remote controls nearby.                                                                                                                                                                                                                                           |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `clean-triage-pass` / `clean`     | clean                                | Here's a brief description of the image: The image shows two tabby cats lying side-by-side on a bright pink surface, likely a couch cushion or blanket. Both cats are relaxed and appear to be sleeping. Each ca ... [tail] ur patterns and are in a comfortable, stretched-out position. The overall scene is cute and playful.   |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `clean-triage-pass` / `clean`     | clean                                | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and is laying on its side with its head facing the right side of the image. The cat on the righ ... [tail] mote control on the left side of the image and a gray remote control on the right side of the image.   |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `clean-triage-pass` / `clean`     | clean                                | The image features two cats lying on a pink surface, possibly a couch or bed. The cats are positioned in a relaxed manner, with one cat lying on its back and the other on its side. Both cats have a tabby patt ... [tail] ting. The overall scene conveys a sense of comfort and leisure, typical of a cat's resting behavior.   |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `clean-triage-pass` / `clean`     | clean                                | The image shows two tabby cats resting on a bright pink blanket spread over a red couch. The smaller cat (left) lies stretched out with a green collar, while the larger cat (right) curls slightly. Two remote ... [tail] o tabby cats rest on a pink blanket over a red couch, with two remote controls nearby.<\|end_of_box\|>  |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`       | cutoff; degeneration                 | Okay, let's break this down. The user wants a brief description of an image, and they've given me a real head-scratcher of a prompt. "Red cat, red sofa, remote control, tabby cat, sleeping, on the red sofa, w ... [tail] ription of a picture with a red cat, a red sofa, a remote control, and a tabby cat, both sleeping on   |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `clean-triage-pass` / `clean`     | clean                                | Here's a brief description of the image: The image shows two tabby kittens lying on their backs on a bright pink surface (likely a couch or blanket). Both kittens are relaxed and appear to be sleeping. Each k ... [tail] ol resting on its belly. The kittens have striped fur patterns and are in a playful, endearing pose.   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `avoid` / `cutoff degraded`       | cutoff                               | In the tranquil setting of this image, two feline companions, one adult and one kitten, are enjoying a peaceful slumber on a vibrant red couch. The adult cat, with its coat of brown and black stripes, is curl ... [tail] ence suggesting recent human activity. The remote controls, along with the cats, create a harmonious   |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black stripes. Both cats are curled up with their tails tucked in. Two remote controls are visible on the couch behind them.                                                                 |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `clean-triage-pass` / `clean`     | clean                                | A top-down view of two cats sleeping on a pink blanket. The cat on the left is a gray and black tabby cat, and it is lying on its side with its head facing the right and its front paws stretched out in front ... [tail] Its tail is sticking out to the left. There are two white and blue remote controls between the cats.    |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black stripes. Both cats are curled up with their tails tucked in. Two remote controls are visible on the couch behind them.                                                                 |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `cutoff degraded`       | cutoff; reasoning-leak               | Here are my reasoning steps: The user asks: "Describe this image briefly." The image shows two cats sleeping on a pink couch, with a remote control and a DVD case (or something) on the couch. The cats are bot ... [tail] a DVD case (or a book) near them. The couch is pink. The cats have striped fur. The remote is white    |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `cutoff degraded`       | cutoff                               | In the tranquil setting of this image, two feline companions, one a tabby cat and the other a Bengal cat, are enjoying a peaceful slumber on a vibrant red couch. The tabby cat, with its distinctive striped co ... [tail] th to the scene. The image captures a moment of calm and companionship between two cats, set against   |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `clean-triage-pass` / `clean`     | clean                                | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a kitten with a fluffy tail and a green collar, while the cat on the ri ... [tail] appears to be a playful and cozy scene, with the cats seemingly enjoying a relaxing moment together.   |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `clean-triage-pass` / `clean`     | clean                                | This image captures a cozy scene of two cats lounging on a pink blanket. The cats are positioned side by side, with one on the left and one on the right. Both felines have similar fur patterns, featuring a mi ... [tail] warm, inviting setting. This image beautifully captures a moment of feline tranquility and comfort.    |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `clean-triage-pass` / `clean`     | clean                                | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and a black tail. It is laying on its side with its head on the pink blanket and its body facin ... [tail] mote control on the left side of the image and a gray remote control on the right side of the image.   |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`       | cutoff; reasoning-leak               | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing the other cat, with a remote control near its head. The cat on t ... [tail] e other cat, with a light - colored remote control near its head. The cat on the right is positioned   |
<!-- markdownlint-enable MD034 -->

## All Model Output and Cost Summary

Every model in this run, with its skim-first output or diagnostic beside the main runtime and peak-memory signals.

<!-- markdownlint-disable MD034 -->

| Model                                                                                                                   | Result                            | Output / diagnostic                                                                                                                                                                                                            | Gen tok   | Total   | Gen TPS   | Peak GB   | Quality signal                       |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|---------|-----------|-----------|--------------------------------------|
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `avoid` / `semantic mismatch`     | [mlx-vlm; model-error] Error: Model Error - Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution ... [tail] y (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).    | -         | 165.28s | -         | -         | mlx-vlm; model-error                 |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `clean-triage-pass` / `clean`     | Two cats are laying on a pink couch.                                                                                                                                                                                           | 10        | 0.42s   | 532       | 1.0       | clean                                |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket.                                                                                                                                                                                       | 10        | 0.64s   | 320       | 3.0       | clean                                |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink couch next to two remote controls.                                                                                                                                                             | 15        | 0.79s   | 304       | 2.1       | clean                                |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                            | 13        | 0.81s   | 190       | 4.1       | clean                                |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `clean-triage-pass` / `clean`     | The image shows two cats lying on a pink blanket, with remote controls nearby.                                                                                                                                                 | 17        | 1.13s   | 267       | 3.0       | clean                                |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `clean-triage-pass` / `clean`     | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats have striped coats, and they are positioned on a cou ... [tail] re of the cats. There are no texts or other objects in the image.   | 81        | 0.89s   | 374       | 1.9       | clean                                |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `clean-triage-pass` / `clean`     | The image shows two cats lying on a pink blanket. One cat is on the left side, while the other is on the right side. There are two remote contr ... [tail] he blanket, one on the left side and the other on the right side.   | 52        | 0.83s   | 329       | 2.5       | clean                                |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `clean-triage-pass` / `clean`     | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                                                             | 35        | 1.00s   | 116       | 4.2       | clean                                |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                            | 13        | 1.22s   | 129       | 5.5       | clean                                |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                            | 13        | 1.26s   | 124       | 5.5       | clean                                |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                        | 19        | 1.49s   | 58.9      | 9.2       | clean                                |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                        | 19        | 1.57s   | 57.5      | 9.2       | clean                                |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `clean-triage-pass` / `clean`     | This image depicts two cats lying on a soft, pink cushion or blanket. - The cat on the left appears to be a kitten, with a smaller size, stripe ... [tail] t looks like a remote control, suggesting a cozy, indoor setting.   | 97        | 1.74s   | 203       | 4.5       | clean                                |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `clean-triage-pass` / `clean`     | Two tabby cats are lying on a pink blanket on a red couch, with two remote controls nearby.                                                                                                                                    | 21        | 3.47s   | 62.6      | 17        | clean                                |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `clean-triage-pass` / `clean`     | This is a close-up photograph of two cats sleeping on a bright pink couch. The cat on the left is a tabby with a black and grey pattern, lying ... [tail] mote controls on the couch, one white and one with a blue button.    | 89        | 1.59s   | 133       | 5.3       | clean                                |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `clean-triage-pass` / `clean`     | This is a close-up photograph of two cats sleeping on a bright pink couch. The cat on the left is a tabby with a black and grey pattern, lying ... [tail] mote controls on the couch, one white and one with a blue button.    | 89        | 1.48s   | 133       | 5.2       | clean                                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `clean-triage-pass` / `clean`     | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls. The cat on the left is curled with its tail ... [tail] ies stretched out, both appearing deeply relaxed and comfortable.   | 51        | 3.26s   | 118       | 21        | clean                                |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `clean-triage-pass` / `clean`     | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled with its head down. Two remote controls are also visible on the couch.                                     | 42        | 2.12s   | 83.9      | 7.7       | clean                                |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `clean-triage-pass` / `clean`     | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray. The cat on the lef ... [tail] hed out, both appearing deeply relaxed in a cozy, domestic scene.   | 61        | 4.20s   | 99.5      | 30        | clean                                |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `clean-triage-pass` / `clean`     | This image shows two tabby cats sleeping on a bright pink couch. The cat on the left is a smaller kitten, lying on its side with its head resti ... [tail] and comfortable, curled up on the soft pink fabric of the couch.    | 75        | 2.51s   | 95.4      | 7.0       | clean                                |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `caveat` / `context budget`       | Cat.                                                                                                                                                                                                                           | 3         | 2.30s   | 84.5      | 4.6       | harness:long-context; context-budget |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `clean-triage-pass` / `clean`     | Two tabby cats are sleeping on a pink blanket on a red couch next to remote controls.                                                                                                                                          | 19        | 4.38s   | 24.4      | 28        | clean                                |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `clean-triage-pass` / `clean`     | A high-angle shot shows two tabby cats sleeping on a pink blanket next to two remote controls.                                                                                                                                 | 21        | 4.61s   | 27.5      | 29        | clean                                |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `clean-triage-pass` / `clean`     | The image shows two tabby cats lying on a pink couch. Both cats are stretched out, appearing relaxed. Between them are two remote controls. The ... [tail] and white stripes. The scene conveys a cozy and calm atmosphere.    | 65        | 2.40s   | 66.6      | 10        | clean                                |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `clean-triage-pass` / `clean`     | The image shows two tabby cats resting on a bright pink couch. One cat lies on its side, eyes closed, with a remote control nearby. The other c ... [tail] appearing relaxed. The scene conveys a cozy, peaceful atmosphere.   | 58        | 2.79s   | 59.0      | 10        | clean                                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `clean-triage-pass` / `clean`     | This image shows two tabby cats lounging on a pink cushion. Both cats appear relaxed, lying on their sides and stretching out their paws. One c ... [tail] ther faces right, and there's a remote control near each of them.   | 54        | 2.98s   | 66.8      | 10        | clean                                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `clean-triage-pass` / `clean`     | This image shows two cats lying on their backs on a pink blanket or cushion. Both cats appear relaxed and are positioned close to each other, w ... [tail] em, suggesting a cozy indoor setting, possibly on a couch or bed.   | 60        | 2.99s   | 70.2      | 9.8       | clean                                |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `harness`               | So, let's see. The image shows two cats lying on a pink couch. The couch is a bright pink color, and there are two remote controls next to the ... [tail] ur and the vibrant pink fabric of the couch forming the backdrop.    | 188       | 2.22s   | 134       | 5.3       | harness:stop-token                   |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `clean-triage-pass` / `clean`     | In this image we can see two cats on the sofa. There are two remotes on the sofa.<end_of_utterance>                                                                                                                            | 23        | 3.78s   | 33.7      | 19        | formatting                           |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `clean-triage-pass` / `token cap` | In the image, a cat is comfortably nestled on a vibrant red couch. The feline, with its fur a mix of black and white, is facing the camera, its ... [tail] presence of the remote control. It's a snapshot of everyday life,   | 200       | 2.48s   | 132       | 5.5       | hit token cap (200)                  |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`       | have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this ima...   | 200       | 3.30s   | 123       | 6.0       | repetitive; cutoff                   |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `clean-triage-pass` / `clean`     | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                                                              | 25        | 4.59s   | 21.0      | 20        | clean                                |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `clean-triage-pass` / `clean`     | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                                                              | 25        | 3.98s   | 22.7      | 15        | clean                                |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `clean-triage-pass` / `clean`     | Two tabby cats are lounging on a bright pink couch, each with a remote control nearby. The cat on the left is stretched out on its back, paws i ... [tail] h appear relaxed and comfortable in their cozy, colorful setting.   | 64        | 4.62s   | 33.6      | 19        | clean                                |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `clean-triage-pass` / `clean`     | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing ... [tail] cene has a high-contrast, stylized visual effect.<\|im_assistant\|> | 167       | 4.49s   | 77.9      | 16        | reasoning-leak                       |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `clean-triage-pass` / `clean`     | Two tabby cats are sleeping peacefully on a bright pink couch, each nestled beside a remote control — one white, one with colorful buttons — creating a cozy and slightly humorous scene of feline comfort.                    | 41        | 9.39s   | 61.8      | 71        | clean                                |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `clean-triage-pass` / `clean`     | The image shows two cats sleeping on a pink blanket. One cat is a small kitten, and the other is a larger adult cat. Both are lying on their sides, and there are two remote controls placed near them on the couch.           | 48        | 4.38s   | 33.9      | 18        | clean                                |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `clean-triage-pass` / `clean`     | The image shows two cats lying on a pink blanket. The cat on the left is a kitten with a black and white striped tail. The cat on the right is a tabby cat with a black and brown striped tail. Both cats are sleeping.        | 55        | 3.66s   | 63.8      | 9.7       | clean                                |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `clean-triage-pass` / `clean`     | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray — creating a cozy, domestic scene.                                                                 | 35        | 9.16s   | 70.5      | 71        | clean                                |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `clean-triage-pass` / `clean`     | Two tabby cats are sleeping peacefully on a bright pink couch. One cat is stretched out on its side, while the other is curled up nearby. Two remote controls lie on the couch between them.                                   | 41        | 5.84s   | 19.5      | 30        | clean                                |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `clean-triage-pass` / `clean`     | Two tabby cats are peacefully napping on a bright pink couch. The cat on the left is lying on its back with its paws curled up, its tail curled ... [tail] The cats appear relaxed and content, enjoying a comfortable nap.    | 124       | 5.21s   | 48.4      | 17        | clean                                |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`       | ◁think▷Okay, let me try to figure out how to describe this image. First, I need to look at the image carefully. There are two cats on a pink su ... [tail] e comfortable with each other. The remotes suggest a home setting   | 200       | 5.31s   | 73.9      | 20        | cutoff; reasoning-leak               |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `clean-triage-pass` / `clean`     | Two tabby cats are sleeping peacefully on a bright pink couch. One cat lies on its side near a white remote control, while the other stretches ... [tail] a remote. Both appear relaxed and comfortable in their cozy spot.    | 49        | 6.32s   | 19.4      | 30        | clean                                |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a red couch, with two remote controls nearby.                                                                                                                                       | 20        | 5.07s   | 32.7      | 19        | clean                                |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `clean-triage-pass` / `clean`     | Here's a brief description of the image: The image shows two tabby cats lying side-by-side on a bright pink surface, likely a couch cushion or ... [tail] e, stretched-out position. The overall scene is cute and playful.    | 91        | 5.87s   | 31.8      | 18        | clean                                |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `clean-triage-pass` / `clean`     | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and is laying on its side with ... [tail] e image and a gray remote control on the right side of the image.   | 101       | 5.25s   | 35.9      | 11        | clean                                |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `clean-triage-pass` / `clean`     | The image features two cats lying on a pink surface, possibly a couch or bed. The cats are positioned in a relaxed manner, with one cat lying o ... [tail] ense of comfort and leisure, typical of a cat's resting behavior.   | 92        | 6.31s   | 27.1      | 20        | clean                                |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `clean-triage-pass` / `clean`     | The image shows two tabby cats resting on a bright pink blanket spread over a red couch. The smaller cat (left) lies stretched out with a green ... [tail] over a red couch, with two remote controls nearby.<\|end_of_box\|>  | 91        | 9.76s   | 42.9      | 63        | clean                                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`       | Okay, let's break this down. The user wants a brief description of an image, and they've given me a real head-scratcher of a prompt. "Red cat, ... [tail] , a red sofa, a remote control, and a tabby cat, both sleeping on    | 200       | 9.18s   | 59.2      | 60        | cutoff; degeneration                 |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `clean-triage-pass` / `clean`     | Here's a brief description of the image: The image shows two tabby kittens lying on their backs on a bright pink surface (likely a couch or bla ... [tail] s have striped fur patterns and are in a playful, endearing pose.   | 70        | 8.11s   | 17.8      | 32        | clean                                |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `avoid` / `cutoff degraded`       | In the tranquil setting of this image, two feline companions, one adult and one kitten, are enjoying a peaceful slumber on a vibrant red couch. ... [tail] ty. The remote controls, along with the cats, create a harmonious   | 200       | 7.46s   | 39.5      | 15        | cutoff                               |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black str ... [tail] ked in. Two remote controls are visible on the couch behind them.   | 55        | 7.08s   | 52.1      | 20        | clean                                |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `clean-triage-pass` / `clean`     | A top-down view of two cats sleeping on a pink blanket. The cat on the left is a gray and black tabby cat, and it is lying on its side with its ... [tail] t. There are two white and blue remote controls between the cats.   | 119       | 7.89s   | 19.8      | 10        | clean                                |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black str ... [tail] ked in. Two remote controls are visible on the couch behind them.   | 55        | 8.57s   | 30.4      | 27        | clean                                |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `cutoff degraded`       | Here are my reasoning steps: The user asks: "Describe this image briefly." The image shows two cats sleeping on a pink couch, with a remote con ... [tail] The couch is pink. The cats have striped fur. The remote is white   | 200       | 8.63s   | 34.4      | 14        | cutoff; reasoning-leak               |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `cutoff degraded`       | In the tranquil setting of this image, two feline companions, one a tabby cat and the other a Bengal cat, are enjoying a peaceful slumber on a ... [tail] a moment of calm and companionship between two cats, set against     | 200       | 13.37s  | 20.0      | 27        | cutoff                               |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `clean-triage-pass` / `clean`     | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a kitt ... [tail] ene, with the cats seemingly enjoying a relaxing moment together.   | 77        | 22.76s  | 4.14      | 25        | clean                                |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `clean-triage-pass` / `clean`     | This image captures a cozy scene of two cats lounging on a pink blanket. The cats are positioned side by side, with one on the left and one on ... [tail] beautifully captures a moment of feline tranquility and comfort.     | 163       | 30.57s  | 5.93      | 23        | clean                                |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `clean-triage-pass` / `clean`     | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and a black tail. It is laying ... [tail] e image and a gray remote control on the right side of the image.   | 159       | 33.01s  | 5.41      | 26        | clean                                |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`       | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing ... [tail] remote control near its head. The cat on the right is positioned    | 200       | 47.46s  | 4.64      | 39        | cutoff; reasoning-leak               |
<!-- markdownlint-enable MD034 -->

## Quick Navigation

- _Successful outputs:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava), +52 more
- _Flagged outputs:_ [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16),
  [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit),
  [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx),
  [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +5 more
- _Failed outputs:_ [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ❌ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `model`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_DECODE_MODEL`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Error summary:_ see error details below
- _Key signals:_ text-sanity=gibberish(char_noise)
- _Failure context:_ type `ValueError`; phase `decode`; code
  `MLX_VLM_DECODE_MODEL`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text 6 tok; estimated non-text n/a;
  generated n/a; requested max 200 tok; stop reason exception

_Error details:_

> Model runtime error during generation for mlx-community/gemma-4-31b-bf16:
> [METAL] Command buffer execution failed: Insufficient Memory
> (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/common.py", line 141, in wired_limit
    yield
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1075, in stream_generate
    detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 171, in add_token
    if self.is_byte_token[token]:
       ~~~~~~~~~~~~~~~~~~^^^^^^^
IndexError: list index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20109, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20485, in _generate_once
    return strict_generate(
        model=model,
    ...<21 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1242, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1009, in stream_generate
    with wired_limit(model, [generation_stream]):
         ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/common.py", line 145, in wired_limit
    mx.synchronize(stream)
    ~~~~~~~~~~~~~~^^^^^^^^
RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20669, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20516, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20143, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
```

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.35s; Gen 0.06s; Total 0.42s
- _Throughput:_ Prompt 3,459 TPS (80 tok); Gen 532 TPS (10 tok)
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
- _Timing:_ Load 0.52s; Gen 0.12s; Total 0.64s
- _Throughput:_ Prompt 4,141 TPS (269 tok); Gen 320 TPS (10 tok)
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 10 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.62s; Gen 0.16s; Total 0.79s
- _Throughput:_ Prompt 321 TPS (26 tok); Gen 304 TPS (15 tok)
- _Tokens:_ prompt 26 tok; estimated text 6 tok; estimated non-text 20 tok;
  generated 15 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch next to two remote controls.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.64s; Gen 0.16s; Total 0.81s
- _Throughput:_ Prompt 3,894 TPS (269 tok); Gen 190 TPS (13 tok)
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
- _Timing:_ Load 0.91s; Gen 0.20s; Total 1.13s
- _Throughput:_ Prompt 2,178 TPS (228 tok); Gen 267 TPS (17 tok)
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
- _Timing:_ Load 0.59s; Gen 0.30s; Total 0.89s
- _Throughput:_ Prompt 319 TPS (22 tok); Gen 374 TPS (81 tok)
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
- _Timing:_ Load 0.50s; Gen 0.32s; Total 0.83s
- _Throughput:_ Prompt 2,895 TPS (417 tok); Gen 329 TPS (52 tok)
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
- _Timing:_ Load 0.60s; Gen 0.40s; Total 1.00s
- _Throughput:_ Prompt 247 TPS (22 tok); Gen 116 TPS (35 tok)
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 35 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image features two cats lying on a couch. One cat is a light brown and
> the other is a dark brown. They both have green eyes and a black nose.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.69s; Gen 0.52s; Total 1.22s
- _Throughput:_ Prompt 3,436 TPS (1,196 tok); Gen 129 TPS (13 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.72s; Gen 0.53s; Total 1.26s
- _Throughput:_ Prompt 3,168 TPS (1,196 tok); Gen 124 TPS (13 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.90s; Gen 0.59s; Total 1.49s
- _Throughput:_ Prompt 3,832 TPS (770 tok); Gen 58.9 TPS (19 tok)
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
- _Timing:_ Load 0.96s; Gen 0.61s; Total 1.57s
- _Throughput:_ Prompt 2,953 TPS (770 tok); Gen 57.5 TPS (19 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 19 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch with remote controls beside them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.97s; Gen 0.75s; Total 1.74s
- _Throughput:_ Prompt 3,994 TPS (973 tok); Gen 203 TPS (97 tok)
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

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.68s; Gen 0.78s; Total 3.47s
- _Throughput:_ Prompt 800 TPS (286 tok); Gen 62.6 TPS (21 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 21 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are lying on a pink blanket on a red couch, with two remote
> controls nearby.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Instruct-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.80s; Gen 0.79s; Total 1.59s
- _Throughput:_ Prompt 3,683 TPS (315 tok); Gen 133 TPS (89 tok)
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
- _Timing:_ Load 0.68s; Gen 0.79s; Total 1.48s
- _Throughput:_ Prompt 3,643 TPS (315 tok); Gen 133 TPS (89 tok)
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

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.44s; Gen 0.80s; Total 3.26s
- _Throughput:_ Prompt 880 TPS (319 tok); Gen 118 TPS (51 tok)
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

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.29s; Gen 0.82s; Total 2.12s
- _Throughput:_ Prompt 1,313 TPS (408 tok); Gen 83.9 TPS (42 tok)
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

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.13s; Gen 1.07s; Total 4.20s
- _Throughput:_ Prompt 723 TPS (319 tok); Gen 99.5 TPS (61 tok)
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

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.43s; Gen 1.07s; Total 2.51s
- _Throughput:_ Prompt 1,181 TPS (319 tok); Gen 95.4 TPS (75 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 75 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two tabby cats sleeping on a bright pink couch. The cat on
> the left is a smaller kitten, lying on its side with its head resting near a
> white remote control. The cat on the right is larger and also sleeping,
> positioned next to another remote control. Both cats appear relaxed and
> comfortable, curled up on the soft pink fabric of the couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 3 tokens.; At long prompt
  length (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%;
  weak text signal truncated).; output/prompt=0.07%; nontext prompt
  burden=100%
- _Timing:_ Load 1.17s; Gen 1.12s; Total 2.30s
- _Throughput:_ Prompt 3,858 TPS (4,103 tok); Gen 84.5 TPS (3 tok)
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

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-mxfp8

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.14s; Gen 1.23s; Total 4.38s
- _Throughput:_ Prompt 986 TPS (286 tok); Gen 24.4 TPS (19 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 19 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping on a pink blanket on a red couch next to remote
> controls.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.33s; Gen 1.25s; Total 4.61s
- _Throughput:_ Prompt 932 TPS (286 tok); Gen 27.5 TPS (21 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 21 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A high-angle shot shows two tabby cats sleeping on a pink blanket next to
> two remote controls.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.12s; Gen 1.27s; Total 2.40s
- _Throughput:_ Prompt 1,785 TPS (417 tok); Gen 66.6 TPS (65 tok)
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

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.42s; Gen 1.36s; Total 2.79s
- _Throughput:_ Prompt 1,124 TPS (408 tok); Gen 59.0 TPS (58 tok)
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 58 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats resting on a bright pink couch. One cat lies
> on its side, eyes closed, with a remote control nearby. The other cat,
> larger and more robust, is curled slightly, also appearing relaxed. The
> scene conveys a cozy, peaceful atmosphere.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.50s; Gen 1.47s; Total 2.98s
- _Throughput:_ Prompt 1,498 TPS (974 tok); Gen 66.8 TPS (54 tok)
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
- _Timing:_ Load 1.47s; Gen 1.51s; Total 2.99s
- _Throughput:_ Prompt 1,514 TPS (974 tok); Gen 70.2 TPS (60 tok)
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

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; text-sanity=gibberish(token_noise)
- _Timing:_ Load 0.69s; Gen 1.52s; Total 2.22s
- _Throughput:_ Prompt 3,766 TPS (317 tok); Gen 134 TPS (188 tok)
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

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ formatting=Unknown tags: &lt;end_of_utterance&gt;
- _Timing:_ Load 1.95s; Gen 1.82s; Total 3.78s
- _Throughput:_ Prompt 2,127 TPS (2,327 tok); Gen 33.7 TPS (23 tok)
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

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ clean triage pass; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200)
- _Timing:_ Load 0.61s; Gen 1.86s; Total 2.48s
- _Throughput:_ Prompt 348 TPS (97 tok); Gen 132 TPS (200 tok)
- _Tokens:_ prompt 97 tok; estimated text 6 tok; estimated non-text 91 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In the image, a cat is comfortably nestled on a vibrant red couch. The
> feline, with its fur a mix of black and white, is facing the camera, its
> eyes half-closed in what appears to be a state of relaxation. The cat's body
> is stretched out, with its front paws neatly tucked under its body and its
> hind legs stretched out behind it.
>
> The couch on which the cat is resting is a striking shade of red, providing
> a vivid contrast to the cat's fur. The couch is positioned against a wall,
> which is painted in a soothing shade of blue.
>
> On the left side of the couch, there's a white remote control, its buttons
> and slots clearly visible. It's as if the remote control has been left
> behind, perhaps forgotten in the hustle and bustle of daily life.
>
> The image captures a moment of tranquility, with the cat seemingly
> undisturbed by the presence of the remote control. It's a snapshot of
> everyday life,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); repetitive token=phrase: "have this
  image. have..."
- _Timing:_ Load 1.42s; Gen 1.87s; Total 3.30s
- _Throughput:_ Prompt 1,134 TPS (266 tok); Gen 123 TPS (200 tok)
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

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.61s; Gen 1.96s; Total 4.59s
- _Throughput:_ Prompt 377 TPS (286 tok); Gen 21.0 TPS (25 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping on a bright pink blanket on a red couch, with
> two remote controls lying next to them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.50s; Gen 2.46s; Total 3.98s
- _Throughput:_ Prompt 11.9 TPS (16 tok); Gen 22.7 TPS (25 tok)
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats lying on a pink blanket, with two remote
> controls placed on the couch behind them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.08s; Gen 2.52s; Total 4.62s
- _Throughput:_ Prompt 530 TPS (319 tok); Gen 33.6 TPS (64 tok)
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

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ reasoning leak
- _Timing:_ Load 1.95s; Gen 2.53s; Total 4.49s
- _Throughput:_ Prompt 1,400 TPS (399 tok); Gen 77.9 TPS (167 tok)
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

- Reasoning leak (◁think▷, ◁/think▷)

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### ✅ mlx-community/Ornith-1.0-35B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 6.70s; Gen 2.66s; Total 9.39s
- _Throughput:_ Prompt 161 TPS (319 tok); Gen 61.8 TPS (41 tok)
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

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.61s; Gen 2.75s; Total 4.38s
- _Throughput:_ Prompt 2,558 TPS (3,341 tok); Gen 33.9 TPS (48 tok)
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

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.90s; Gen 2.75s; Total 3.66s
- _Throughput:_ Prompt 1,260 TPS (2,356 tok); Gen 63.8 TPS (55 tok)
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

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 6.39s; Gen 2.76s; Total 9.16s
- _Throughput:_ Prompt 142 TPS (319 tok); Gen 70.5 TPS (35 tok)
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

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.02s; Gen 2.80s; Total 5.84s
- _Throughput:_ Prompt 471 TPS (319 tok); Gen 19.5 TPS (41 tok)
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
- _Timing:_ Load 2.29s; Gen 2.90s; Total 5.21s
- _Throughput:_ Prompt 831 TPS (274 tok); Gen 48.4 TPS (124 tok)
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

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); reasoning leak
- _Timing:_ Load 2.18s; Gen 3.12s; Total 5.31s
- _Throughput:_ Prompt 1,370 TPS (399 tok); Gen 73.9 TPS (200 tok)
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

- Reasoning leak (◁think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.09s; Gen 3.21s; Total 6.32s
- _Throughput:_ Prompt 474 TPS (319 tok); Gen 19.4 TPS (49 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 49 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping peacefully on a bright pink couch. One cat lies
> on its side near a white remote control, while the other stretches out
> nearby, also next to a remote. Both appear relaxed and comfortable in their
> cozy spot.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.75s; Gen 3.32s; Total 5.07s
- _Throughput:_ Prompt 1,250 TPS (3,341 tok); Gen 32.7 TPS (20 tok)
- _Tokens:_ prompt 3341 tok; estimated text 6 tok; estimated non-text 3335
  tok; generated 20 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a red couch, with two remote
> controls nearby.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.28s; Gen 3.58s; Total 5.87s
- _Throughput:_ Prompt 396 TPS (275 tok); Gen 31.8 TPS (91 tok)
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
- _Timing:_ Load 1.56s; Gen 3.67s; Total 5.25s
- _Throughput:_ Prompt 1,313 TPS (1,031 tok); Gen 35.9 TPS (101 tok)
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
- _Timing:_ Load 2.14s; Gen 4.16s; Total 6.31s
- _Throughput:_ Prompt 587 TPS (441 tok); Gen 27.1 TPS (92 tok)
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

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 5.57s; Gen 4.18s; Total 9.76s
- _Throughput:_ Prompt 200 TPS (408 tok); Gen 42.9 TPS (91 tok)
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
- _Timing:_ Load 4.76s; Gen 4.41s; Total 9.18s
- _Throughput:_ Prompt 431 TPS (439 tok); Gen 59.2 TPS (200 tok)
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

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.31s; Gen 4.79s; Total 8.11s
- _Throughput:_ Prompt 332 TPS (275 tok); Gen 17.8 TPS (70 tok)
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

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200)
- _Timing:_ Load 1.66s; Gen 5.79s; Total 7.46s
- _Throughput:_ Prompt 1,742 TPS (1,239 tok); Gen 39.5 TPS (200 tok)
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

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.18s; Gen 5.89s; Total 7.08s
- _Throughput:_ Prompt 155 TPS (745 tok); Gen 52.1 TPS (55 tok)
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

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.48s; Gen 6.39s; Total 7.89s
- _Throughput:_ Prompt 2,979 TPS (1,031 tok); Gen 19.8 TPS (119 tok)
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

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.71s; Gen 6.85s; Total 8.57s
- _Throughput:_ Prompt 148 TPS (745 tok); Gen 30.4 TPS (55 tok)
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

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); reasoning leak
- _Timing:_ Load 1.64s; Gen 6.98s; Total 8.63s
- _Throughput:_ Prompt 1,150 TPS (1,330 tok); Gen 34.4 TPS (200 tok)
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

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200)
- _Timing:_ Load 2.57s; Gen 10.80s; Total 13.37s
- _Throughput:_ Prompt 1,593 TPS (1,239 tok); Gen 20.0 TPS (200 tok)
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
- _Timing:_ Load 2.17s; Gen 20.58s; Total 22.76s
- _Throughput:_ Prompt 8.77 TPS (17 tok); Gen 4.14 TPS (77 tok)
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
- _Timing:_ Load 2.20s; Gen 28.36s; Total 30.57s
- _Throughput:_ Prompt 1,014 TPS (766 tok); Gen 5.93 TPS (163 tok)
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
- _Timing:_ Load 2.48s; Gen 30.52s; Total 33.01s
- _Throughput:_ Prompt 952 TPS (1,031 tok); Gen 5.41 TPS (159 tok)
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
- _Key signals:_ hit token cap (200); reasoning leak
- _Timing:_ Load 3.31s; Gen 44.13s; Total 47.46s
- _Throughput:_ Prompt 495 TPS (399 tok); Gen 4.64 TPS (200 tok)
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

- Reasoning leak (◁think▷, ◁/think▷)
- Likely capped by max token budget

---

<!-- markdownlint-enable MD033 MD034 -->
