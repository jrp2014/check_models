# Model Output Gallery

_Generated on 2026-06-20 00:48:07 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## Run Stamps

- `mlx-vlm`: `0.6.3`
- `mlx`: `0.32.0.dev20260620+c9e8ee6b`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.20.1`
- _Python Version:_ 3.13.13
- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.1
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

| Model                                                                                                                   | Result                      | Quality / diagnostic                 | Response / diagnostic                                                                                                                                                                                                                                                                                                                                                         |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------------|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `avoid` / `runtime failure` | mlx; weight-mismatch                 | [mlx; weight-mismatch] Error: Weight Mismatch - Model loading failed: Missing 2 parameters: multi_modal_projector.layer_norm.bias, multi_modal_projector.layer_norm.weight.                                                                                                                                                                                                   |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `avoid` / `runtime failure` | mlx-vlm; model-error                 | [mlx-vlm; model-error] Error: Model Error - Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter                                                                                                                                                                                                                                               |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `recommended` / `clean`     | clean                                | Two cats are laying on a pink couch.                                                                                                                                                                                                                                                                                                                                          |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `recommended` / `clean`     | clean                                | Two cats are sleeping on a pink blanket.                                                                                                                                                                                                                                                                                                                                      |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `recommended` / `clean`     | clean                                | Two cats are sleeping on a pink couch next to two remote controls.                                                                                                                                                                                                                                                                                                            |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `recommended` / `clean`     | clean                                | The image shows two cats lying on a pink blanket. There are remote controls on the blanket as well.                                                                                                                                                                                                                                                                           |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `recommended` / `clean`     | clean                                | The image shows two cats lying on a pink blanket. One cat is on the left side, while the other is on the right side. There are two remote controls placed on the blanket, one on the left side and the other on the right side.                                                                                                                                               |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `recommended` / `clean`     | clean                                | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                                                                                                                                                                                                            |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `recommended` / `clean`     | clean                                | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats have striped coats, and they are positioned on a couch. The cats are facing the camera, and their eyes are open, indi ... [tail] , and it captures the relaxed posture of the cats. There are no texts or other objects in the image.                                              |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `avoid` / `harness`         | harness:prompt-template              | [harness:prompt-template] Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                                 |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `recommended` / `clean`     | clean                                | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                                                                                                                       |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `avoid` / `harness`         | harness:prompt-template              | [harness:prompt-template] Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                                 |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `recommended` / `clean`     | clean                                | Two tabby cats are lying on a pink blanket on a red couch, with a remote control next to each cat.                                                                                                                                                                                                                                                                            |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `recommended` / `clean`     | clean                                | In this image, there are two cats lying on a soft, pink cushion or blanket. - The cat on the left appears to be a kitten, with a smaller size, fluffy fur, and a relaxed posture. ...                                                                                                                                                                                         |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `recommended` / `clean`     | clean                                | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled with its head down. Two remote controls are also visible on the couch.                                                                                                                                                                                    |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `recommended` / `clean`     | clean                                | This is a close-up photograph of two cats sleeping on a bright pink couch. The cat on the left is a tabby with a black and brown striped pattern, lying on its back with its eyes closed. The cat on the right i ... [tail] eacefully. There are two remote controls on the couch, one white and one gray, placed near the cats.                                              |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `recommended` / `clean`     | clean                                | This is a close-up photograph of two cats sleeping on a bright pink couch. The cat on the left is a tabby with a black and brown striped pattern, lying on its back with its eyes closed. The cat on the right i ... [tail] eacefully. There are two remote controls on the couch, one white and one gray, placed near the cats.                                              |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `recommended` / `clean`     | clean                                | This image shows two tabby cats sleeping on a bright pink blanket or couch cover. The cat on the left is a smaller kitten, lying on its side with its head resting near a white remote control. The cat on the r ... [tail] oned next to another remote control. Both cats appear relaxed and comfortable in their cozy setting.                                              |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `recommended` / `clean`     | clean                                | Two tabby cats are sprawled out on a bright pink couch, sleeping peacefully. One cat is curled near the left armrest with a remote beside it, while the other lies stretched across the center, next to another remote. The scene is cozy and relaxed, capturing a quiet moment of feline comfort.                                                                            |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `recommended` / `clean`     | clean                                | A high-angle shot shows two tabby cats sleeping on a pink blanket next to two remote controls.                                                                                                                                                                                                                                                                                |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `recommended` / `clean`     | clean                                | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls. The cat on the left is stretched out with its tail curled, while the one on the right is curled up facing downward. Both appear deeply relaxed in a cozy, domestic scene.                                                                                                 |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `recommended` / `clean`     | clean                                | The image shows two tabby cats lying on a pink couch. Both cats are stretched out, appearing relaxed. Between them are two remote controls. The couch has a vibrant pink cover, and the cats' fur has a mix of brown, black, and white stripes. The scene conveys a cozy and calm atmosphere.                                                                                 |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `caveat` / `context budget` | harness:long-context; context-budget | [harness:long-context; context-budget] Cat.                                                                                                                                                                                                                                                                                                                                   |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `recommended` / `clean`     | clean                                | The image shows two tabby cats resting on a bright pink couch. One cat lies on its side, eyes closed, with a remote control nearby. The other cat, larger and more robust, rests with its head down, also near a remote. The scene is cozy, with the vivid pink couch contrasting with the cats’ striped fur.                                                                 |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `recommended` / `clean`     | clean                                | A high-angle shot shows two tabby cats sleeping on a pink blanket on a red couch, with two remote controls lying near them.                                                                                                                                                                                                                                                   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `recommended` / `clean`     | clean                                | This image shows two tabby cats lounging on a pink blanket. Both cats appear relaxed, lying on their sides with their legs stretched out. One cat faces left, while the other faces right, and there seem to be remote controls near them.                                                                                                                                    |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `recommended` / `clean`     | clean                                | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                                                                                                                                                                                                             |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `harness`         | harness:stop-token                   | [harness:stop-token] So, let's see. The image shows two cats lying on a pink couch. The couch is a bright pink color, and there are two remote controls next to the cats. One cat is on the left, with a striped patter ... [tail] cribe this briefly, so focus on the key elements: two cats, pink couch, remotes. &lt;/think&gt; ...                                        |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `recommended` / `clean`     | clean                                | In the image, a cat is comfortably nestled on a red couch, its body relaxed and at ease. The cat's fur is a mix of black and white, with the black spots standing out against the vibrant red of the couch. The ... [tail] ch is positioned against a white wall, which adds a clean and minimalist aesthetic to the scene. ...                                               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `recommended` / `clean`     | clean                                | This image shows two cats lying on their backs on a pink blanket or cushion. Both cats appear relaxed and playful, with their legs stretched out and paws in the air. One cat is smaller, possibly a kitten, while the other is larger, likely an adult. There are two remote controls placed near them on the blanket.                                                       |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `recommended` / `clean`     | formatting                           | [formatting] In this image we can see two cats on the sofa. There are two remotes on the sofa.<end_of_utterance>                                                                                                                                                                                                                                                              |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded` | repetitive; cutoff                   | [repetitive; cutoff] have this image. ...                                                                                                                                                                                                                                                                                                                                     |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `recommended` / `clean`     | clean                                | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray. The cat on the left is curled with its tail tucked, while the one on the right lies stretched out, both looking deeply relaxed.                                                                                                                  |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `recommended` / `clean`     | clean                                | Two tabby cats are lounging on a bright pink couch, each with a remote control nearby. The cat on the left is stretched out on its back, paws relaxed, while the one on the right lies curled up, head resting near its front paws. Both appear to be napping or resting comfortably in a cozy, playful setting.                                                              |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `recommended` / `clean`     | clean                                | Two tabby cats are sleeping on a bright pink couch. One cat is stretched out on its side, while the other is curled up nearby. Two remote controls lie on the couch between them.                                                                                                                                                                                             |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `recommended` / `clean`     | clean                                | Two tabby cats are peacefully napping on a bright pink couch. The cat on the left is lying on its back with its paws curled up, its tail curled around its body. It has a green collar. The cat on the right is ... [tail] d warm, creating a cozy atmosphere. The cats appear relaxed and content, enjoying a comfortable nap.                                               |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `recommended` / `clean`     | clean                                | The image shows two cats sleeping on a pink blanket. One cat is a small kitten, and the other is a larger adult cat. Both are lying on their sides, and there are two remote controls placed near them on the couch.                                                                                                                                                          |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `recommended` / `clean`     | clean                                | The image shows two tabby cats resting on a bright pink blanket spread over a red couch. The cat on the is stretched out, sleeping with its head down, while the cat on the left lies curled, also asleep. Two remote controls (one gray, one white) rest near the cats, adding a cozy, domestic feel to the scene.                                                           |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `recommended` / `clean`     | clean                                | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                                                                                                                                                                                                             |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `recommended` / `token cap` | reasoning-leak                       | [reasoning-leak] ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing the other cat. The cat on the right is lying on its ... [tail] facing the left cat. Both cats have striped fur with a mix of brown, black, and white patterns.                                               |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `recommended` / `clean`     | clean                                | Two tabby cats are sleeping peacefully on a bright pink couch. One cat lies on its side near a white remote control, while the other stretches out nearby, also next to a remote. Both appear relaxed and comfortable in their cozy spot.                                                                                                                                     |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `avoid` / `harness`         | harness:encoding                     | [harness:encoding] TheĠimageĠfeaturesĠtwoĠcatsĠlyingĠonĠaĠpinkĠsurface,ĠpossiblyĠaĠblanketĠorĠaĠcouch.ĠTheĠcatĠonĠtheĠleftĠisĠaĠkitten,ĠandĠtheĠoneĠonĠtheĠrightĠisĠanĠadultĠcat.ĠBothĠcatsĠareĠinĠrelaxedĠpostures,Ġw ... [tail] controlsĠplacedĠnearĠtheĠcats,ĠoneĠcloserĠtoĠtheĠkittenĠandĠtheĠotherĠcloserĠtoĠtheĠadultĠcat.                                              |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `avoid` / `harness`         | harness:stop-token                   | [harness:stop-token] Two cats are sleeping on a pink couch with remote controls beside them.<\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end... |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `recommended` / `clean`     | clean                                | The image shows two cats lying on a pink blanket. The cat on the left is a kitten with a black and white striped tail. The cat on the right is a tabby cat with a black and brown striped tail. Both cats are sleeping.                                                                                                                                                       |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded` | cutoff; reasoning-leak               | [cutoff; reasoning-leak] ◁think▷Okay, let me try to figure out how to describe this image. First, I need to look at the image carefully. There are two cats on a pink surface, maybe a blanket or a bed. Both cats seem ... [tail] gs or just two cats that are comfortable with each other. The remotes suggest a home setting                                               |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `recommended` / `clean`     | clean                                | Here's a brief description of the image: The image shows two tabby cats lying side-by-side on a bright pink surface, likely a couch cushion or blanket. Both cats are relaxed and appear to be sleeping. Each ca ... [tail] ur patterns and are in a comfortable, stretched-out position. The overall scene is cute and playful.                                              |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `recommended` / `clean`     | clean                                | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and is laying on its side with its head facing the right side of the image. The cat on the righ ... [tail] mote control on the left side of the image and a gray remote control on the right side of the image.                                              |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded` | cutoff; degeneration                 | [cutoff; degeneration] Okay, let's break this down. The user wants a brief description of an image, and they've given me a pretty convoluted OCR output. My first step is always to understand the request. "Briefly des ... [tail] trying to make sense of itself. I need to ignore the OCR and focus on the image itself. ...                                               |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `recommended` / `clean`     | clean                                | Two cats are sleeping on a pink blanket with two remote controls nearby.                                                                                                                                                                                                                                                                                                      |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `recommended` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black stripes. Both cats are curled up with their tails tucked in. Two remote controls are visible on the couch behind them.                                                                                                            |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `recommended` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black stripes. Both cats are curled up with their tails tucked in. Two remote controls are visible on the couch behind them.                                                                                                            |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `harness`         | harness:stop-token; reasoning-leak   | [harness:stop-token; reasoning-leak] Here are my reasoning steps: The user asks: "Describe this image briefly." The image is of two cats sleeping on a pink couch, with a remote control. The user wants a brief description ... [tail] te lying nearby." That's it. No need for extra. We'll comply. [BEGIN FINAL RESPONSE] ...                                              |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `recommended` / `clean`     | clean                                | Here's a brief description of the image: The image shows two tabby kittens lying on their backs on a bright pink surface (likely a couch or blanket). Both kittens are relaxed and appear to be sleeping. Each k ... [tail] ol resting on its belly. The kittens have striped fur patterns and are in a playful, endearing pose.                                              |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `recommended` / `clean`     | clean                                | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                                                                                                                                                                                                          |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `avoid` / `cutoff degraded` | cutoff                               | [cutoff] In the tranquil setting of this image, two feline companions, one adult and one kitten, are enjoying a peaceful slumber on a vibrant red couch. The adult cat, with its coat of brown and black stripes, i ... [tail] ten's legs are stretched out in front of it, adding to the overall sense of comfort and ease. ...                                              |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `recommended` / `clean`     | clean                                | A top-down view of two cats sleeping on a pink blanket. The cat on the left is a gray and black tabby cat, and it is lying on its side with its head facing the right and its front paws stretched out in front ... [tail] Its tail is sticking out to the left. There are two white and blue remote controls between the cats.                                               |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `cutoff degraded` | cutoff                               | [cutoff] In the tranquil setting of this image, two feline companions, one a tabby and the other a Bengal, are enjoying a peaceful slumber on a vibrant red couch. The tabby, with its distinctive striped coat, is ... [tail] y. The contrast between the two cats' sleeping positions adds a touch of whimsy to the scene. ...                                              |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `recommended` / `clean`     | clean                                | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a kitten with a fluffy tail and a green collar, while the cat on the ri ... [tail] appears to be a playful and cozy scene, with the cats seemingly enjoying a relaxing moment together.                                              |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `recommended` / `clean`     | clean                                | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and a black tail. It is laying on its side with its head on the pink blanket and its body facin ... [tail] mote control on the left side of the image and a gray remote control on the right side of the image.                                              |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded` | cutoff; reasoning-leak               | [cutoff; reasoning-leak] ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing the other cat, with a remote control near its h ... [tail] cat, with a light - colored remote control near its head. The cat on the right is positioned                                              |
<!-- markdownlint-enable MD034 -->

## Quick Navigation

- _Successful outputs:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct),
  +50 more
- _Flagged outputs:_ [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16),
  [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16),
  [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit),
  [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit),
  +9 more
- _Failed outputs:_ [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ❌ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Weight Mismatch)
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Weight
  Mismatch`; diagnostic code `MLX_MODEL_LOAD_WEIGHT_MISMATCH`
- _Next step:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.
- _Error summary:_ see error details below
- _Key signals:_ weight mismatch; mlx model load weight mismatch
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_MODEL_LOAD_WEIGHT_MISMATCH`; package `mlx`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception

_Error details:_

> Model loading failed: Missing 2 parameters:
> multi_modal_projector.layer_norm.bias,
> multi_modal_projector.layer_norm.weight.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18780, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18169, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 903, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 783, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 659, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 2 parameters:
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18989, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18790, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Missing 2 parameters:
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

</details>

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ❌ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `mlx-vlm`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect prompt-template, stop-token, and decode post-processing
  behavior.
- _Error summary:_ see error details below
- _Key signals:_ model error; mlx vlm model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_VLM_MODEL_LOAD_MODEL`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception

_Error details:_

> Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no
> setter

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18780, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18169, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 903, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 783, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 534, in load_model
    model_config = apply_generation_config_defaults(model_config, config)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 69, in apply_generation_config_defaults
    setattr(model_config, key, config[key])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: property 'eos_token_id' of 'ModelConfig' object has no setter

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18989, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18790, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter
```

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=92%
- _Timing:_ Load 0.39s; Gen 0.05s; Total 0.45s
- _Throughput:_ Prompt 5,175 TPS (80 tok); Gen 446 TPS (10 tok)
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

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 0.57s; Gen 0.13s; Total 0.70s
- _Throughput:_ Prompt 3,936 TPS (269 tok); Gen 262 TPS (10 tok)
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

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=77%
- _Timing:_ Load 0.71s; Gen 0.19s; Total 0.91s
- _Throughput:_ Prompt 256 TPS (26 tok); Gen 251 TPS (15 tok)
- _Tokens:_ prompt 26 tok; estimated text 6 tok; estimated non-text 20 tok;
  generated 15 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch next to two remote controls.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### ✅ mlx-community/MiniCPM-V-4.6-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=97%
- _Timing:_ Load 0.98s; Gen 0.26s; Total 1.27s
- _Throughput:_ Prompt 1,848 TPS (228 tok); Gen 204 TPS (22 tok)
- _Tokens:_ prompt 228 tok; estimated text 6 tok; estimated non-text 222 tok;
  generated 22 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two cats lying on a pink blanket. There are remote controls
> on the blanket as well.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 0.56s; Gen 0.40s; Total 0.97s
- _Throughput:_ Prompt 2,651 TPS (417 tok); Gen 240 TPS (52 tok)
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

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Timing:_ Load 0.58s; Gen 0.51s; Total 1.09s
- _Throughput:_ Prompt 212 TPS (22 tok); Gen 93.0 TPS (35 tok)
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 35 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image features two cats lying on a couch. One cat is a light brown and
> the other is a dark brown. They both have green eyes and a black nose.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Timing:_ Load 0.70s; Gen 0.56s; Total 1.27s
- _Throughput:_ Prompt 252 TPS (22 tok); Gen 176 TPS (81 tok)
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

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (1.1%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=99%
- _Timing:_ Load 0.71s; Gen 0.66s; Total 1.37s
- _Throughput:_ Prompt 2,506 TPS (1,196 tok); Gen 109 TPS (13 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:output_ratio(1.1%)

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 0.93s; Gen 0.68s; Total 1.62s
- _Throughput:_ Prompt 2,630 TPS (770 tok); Gen 52.2 TPS (19 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 19 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch with remote controls beside them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (1.1%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=99%
- _Timing:_ Load 0.65s; Gen 0.73s; Total 1.39s
- _Throughput:_ Prompt 2,314 TPS (1,196 tok); Gen 96.9 TPS (13 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:output_ratio(1.1%)

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 2.68s; Gen 0.74s; Total 3.43s
- _Throughput:_ Prompt 617 TPS (286 tok); Gen 90.3 TPS (24 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 24 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are lying on a pink blanket on a red couch, with a remote
> control next to each cat.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 1.16s; Gen 0.98s; Total 2.16s
- _Throughput:_ Prompt 2,908 TPS (949 tok); Gen 156 TPS (99 tok)
- _Tokens:_ prompt 949 tok; estimated text 6 tok; estimated non-text 943 tok;
  generated 99 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In this image, there are two cats lying on a soft, pink cushion or blanket.
>
> &#45; The cat on the left appears to be a kitten, with a smaller size, fluffy
> fur, and a relaxed posture.
> &#45; The cat on the right is an adult, with a more robust build, striped tabby
> fur, and is stretched out comfortably, looking slightly at the kitten.
>
> Both are positioned near what looks like a remote control, suggesting a cozy
> indoor setting.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 1.95s; Gen 0.98s; Total 2.93s
- _Throughput:_ Prompt 1,093 TPS (408 tok); Gen 71.1 TPS (42 tok)
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

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Instruct-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 0.89s; Gen 1.01s; Total 1.90s
- _Throughput:_ Prompt 3,212 TPS (315 tok); Gen 110 TPS (96 tok)
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 96 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This is a close-up photograph of two cats sleeping on a bright pink couch.
> The cat on the left is a tabby with a black and brown striped pattern, lying
> on its back with its eyes closed. The cat on the right is a calico with a
> mix of orange, black, and white fur, also lying on its back. Both cats are
> resting peacefully. There are two remote controls on the couch, one white
> and one gray, placed near the cats.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 0.73s; Gen 1.02s; Total 1.76s
- _Throughput:_ Prompt 2,858 TPS (315 tok); Gen 111 TPS (96 tok)
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 96 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This is a close-up photograph of two cats sleeping on a bright pink couch.
> The cat on the left is a tabby with a black and brown striped pattern, lying
> on its back with its eyes closed. The cat on the right is a calico with a
> mix of orange, black, and white fur, also lying on its back. Both cats are
> resting peacefully. There are two remote controls on the couch, one white
> and one gray, placed near the cats.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 1.92s; Gen 1.22s; Total 3.16s
- _Throughput:_ Prompt 1,062 TPS (319 tok); Gen 78.2 TPS (71 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 71 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two tabby cats sleeping on a bright pink blanket or couch
> cover. The cat on the left is a smaller kitten, lying on its side with its
> head resting near a white remote control. The cat on the right is larger and
> also sleeping, positioned next to another remote control. Both cats appear
> relaxed and comfortable in their cozy setting.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 2.89s; Gen 1.33s; Total 4.23s
- _Throughput:_ Prompt 754 TPS (319 tok); Gen 70.8 TPS (63 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 63 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sprawled out on a bright pink couch, sleeping peacefully.
> One cat is curled near the left armrest with a remote beside it, while the
> other lies stretched across the center, next to another remote. The scene is
> cozy and relaxed, capturing a quiet moment of feline comfort.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-mxfp8

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 3.45s; Gen 1.46s; Total 4.92s
- _Throughput:_ Prompt 777 TPS (286 tok); Gen 23.1 TPS (21 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 21 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A high-angle shot shows two tabby cats sleeping on a pink blanket next to
> two remote controls.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 3.48s; Gen 1.48s; Total 4.97s
- _Throughput:_ Prompt 624 TPS (319 tok); Gen 60.9 TPS (58 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 58 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sprawled out asleep on a bright pink couch, nestled
> between two remote controls. The cat on the left is stretched out with its
> tail curled, while the one on the right is curled up facing downward. Both
> appear deeply relaxed in a cozy, domestic scene.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 1.17s; Gen 1.54s; Total 2.72s
- _Throughput:_ Prompt 1,439 TPS (417 tok); Gen 54.9 TPS (65 tok)
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

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 3 tokens.; At long prompt
  length (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%).;
  output/prompt=0.07%; nontext prompt burden=100%
- _Timing:_ Load 1.38s; Gen 1.67s; Total 3.06s
- _Throughput:_ Prompt 2,566 TPS (4,103 tok); Gen 68.3 TPS (3 tok)
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
- long_context_low_ratio(0.1%;4103->3)

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 1.75s; Gen 1.71s; Total 3.48s
- _Throughput:_ Prompt 1,049 TPS (408 tok); Gen 52.9 TPS (69 tok)
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 69 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats resting on a bright pink couch. One cat lies
> on its side, eyes closed, with a remote control nearby. The other cat,
> larger and more robust, rests with its head down, also near a remote. The
> scene is cozy, with the vivid pink couch contrasting with the cats’ striped
> fur.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 3.47s; Gen 1.93s; Total 5.42s
- _Throughput:_ Prompt 798 TPS (286 tok); Gen 20.3 TPS (28 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 28 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A high-angle shot shows two tabby cats sleeping on a pink blanket on a red
> couch, with two remote controls lying near them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 1.59s; Gen 1.98s; Total 3.58s
- _Throughput:_ Prompt 1,024 TPS (950 tok); Gen 50.4 TPS (52 tok)
- _Tokens:_ prompt 950 tok; estimated text 6 tok; estimated non-text 944 tok;
  generated 52 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two tabby cats lounging on a pink blanket. Both cats appear
> relaxed, lying on their sides with their legs stretched out. One cat faces
> left, while the other faces right, and there seem to be remote controls near
> them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 2.81s; Gen 2.04s; Total 4.86s
- _Throughput:_ Prompt 309 TPS (286 tok); Gen 22.8 TPS (25 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping on a bright pink blanket on a red couch, with
> two remote controls lying next to them.
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
  text.; nontext prompt burden=98%
- _Timing:_ Load 0.78s; Gen 2.04s; Total 2.84s
- _Throughput:_ Prompt 3,093 TPS (317 tok); Gen 103 TPS (196 tok)
- _Tokens:_ prompt 317 tok; estimated text 6 tok; estimated non-text 311 tok;
  generated 196 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> So, let's see. The image shows two cats lying on a pink couch. The couch is
> a bright pink color, and there are two remote controls next to the cats. One
> cat is on the left, with a striped pattern, and the other is on the right,
> also striped but maybe a bit different. Both cats are relaxed, lying down,
> maybe sleeping. The remotes are white with some buttons, and one has a blue
> button. The background is the pink couch, so the main elements are the two
> cats, the remotes, and the pink fabric. I need to describe this briefly, so
> focus on the key elements: two cats, pink couch, remotes.
> &lt;/think&gt;
>
> Two tabby cats are lying on a bright pink couch, each resting with their
> paws extended. A white remote control with a blue button sits between them,
> and another remote control is positioned near the left cat. The scene is
> calm, with the cats appearing relaxed.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (gibberish(token_noise))
- token_leak:&lt;/think&gt;

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=94%
- _Timing:_ Load 0.57s; Gen 2.15s; Total 2.73s
- _Throughput:_ Prompt 269 TPS (97 tok); Gen 109 TPS (187 tok)
- _Tokens:_ prompt 97 tok; estimated text 6 tok; estimated non-text 91 tok;
  generated 187 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In the image, a cat is comfortably nestled on a red couch, its body relaxed
> and at ease. The cat's fur is a mix of black and white, with the black
> spots standing out against the vibrant red of the couch. The cat's head is
> turned to the side, and its eyes are closed, suggesting it is either
> sleeping or in a state of deep relaxation.
>
> The couch itself is a striking shade of red, providing a vivid contrast to
> the cat's fur. The couch is positioned against a white wall, which adds a
> clean and minimalist aesthetic to the scene.
>
> In the background, there's a white door, slightly ajar, suggesting that the
> room is not completely enclosed. The door's presence adds a sense of
> openness and spaciousness to the image.
>
> Overall, the image captures a peaceful moment of a cat resting on a red
> couch, surrounded by a minimalist and clean environment.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 1.56s; Gen 2.16s; Total 3.73s
- _Throughput:_ Prompt 1,073 TPS (950 tok); Gen 54.2 TPS (68 tok)
- _Tokens:_ prompt 950 tok; estimated text 6 tok; estimated non-text 944 tok;
  generated 68 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two cats lying on their backs on a pink blanket or cushion.
> Both cats appear relaxed and playful, with their legs stretched out and
> paws in the air. One cat is smaller, possibly a kitten, while the other is
> larger, likely an adult. There are two remote controls placed near them on
> the blanket.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%; formatting=Unknown tags:
  &lt;end_of_utterance&gt;
- _Timing:_ Load 2.27s; Gen 2.36s; Total 4.65s
- _Throughput:_ Prompt 1,565 TPS (2,327 tok); Gen 27.8 TPS (23 tok)
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
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; repetitive
  token=phrase: "have this image. have..."
- _Timing:_ Load 1.68s; Gen 2.97s; Total 4.67s
- _Throughput:_ Prompt 1,028 TPS (266 tok); Gen 74.0 TPS (200 tok)
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

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 6.79s; Gen 3.01s; Total 9.81s
- _Throughput:_ Prompt 168 TPS (319 tok); Gen 50.2 TPS (55 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 55 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sprawled out asleep on a bright pink couch, nestled
> between two remote controls — one white and one gray. The cat on the left is
> curled with its tail tucked, while the one on the right lies stretched out,
> both looking deeply relaxed.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 2.56s; Gen 3.39s; Total 5.98s
- _Throughput:_ Prompt 432 TPS (319 tok); Gen 26.9 TPS (71 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 71 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are lounging on a bright pink couch, each with a remote
> control nearby. The cat on the left is stretched out on its back, paws
> relaxed, while the one on the right lies curled up, head resting near its
> front paws. Both appear to be napping or resting comfortably in a cozy,
> playful setting.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 3.45s; Gen 3.44s; Total 6.92s
- _Throughput:_ Prompt 383 TPS (319 tok); Gen 15.4 TPS (40 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 40 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping on a bright pink couch. One cat is stretched out
> on its side, while the other is curled up nearby. Two remote controls lie
> on the couch between them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 2.44s; Gen 3.55s; Total 6.01s
- _Throughput:_ Prompt 737 TPS (274 tok); Gen 39.2 TPS (124 tok)
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

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Timing:_ Load 1.63s; Gen 3.66s; Total 5.30s
- _Throughput:_ Prompt 1,667 TPS (3,341 tok); Gen 29.6 TPS (48 tok)
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

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 5.93s; Gen 3.71s; Total 9.65s
- _Throughput:_ Prompt 216 TPS (408 tok); Gen 39.3 TPS (71 tok)
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 71 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats resting on a bright pink blanket spread over
> a red couch. The cat on the is stretched out, sleeping with its head down,
> while the cat on the left lies curled, also asleep. Two remote controls (one
> gray, one white) rest near the cats, adding a cozy, domestic feel to the
> scene.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=62%
- _Timing:_ Load 1.75s; Gen 3.83s; Total 5.59s
- _Throughput:_ Prompt 8.5 TPS (16 tok); Gen 13.0 TPS (25 tok)
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats lying on a pink blanket, with two remote
> controls placed on the couch behind them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; reasoning
  leak
- _Timing:_ Load 2.37s; Gen 3.89s; Total 6.27s
- _Throughput:_ Prompt 1,200 TPS (399 tok); Gen 58.1 TPS (200 tok)
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's describe the image. First, there are two tabby cats on a
> pink fabric surface. The cat on the left is lying on its side, facing the
> other cat. The cat on the right is lying on its back, facing the left cat.
> Both cats have striped fur with a mix of brown, black, and white. There are
> two remote controls: one near the left cat and one near the right cat. The
> image has a high-contrast, possibly filtered look with enhanced colors,
> making the pink background and the cats' fur stand out. The overall scene is
> cozy, with the cats appearing relaxed.◁/think▷The image shows two tabby
> cats resting on a pink fabric surface. The cat on the left lies on its side,
> facing the other cat, while the cat on the right lies on its back, also
> facing the left cat. Both cats have striped fur with a mix of brown, black,
> and white patterns.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷, ◁/think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 3.48s; Gen 3.95s; Total 7.46s
- _Throughput:_ Prompt 391 TPS (319 tok); Gen 15.7 TPS (49 tok)
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

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 76 occurrences).; nontext prompt burden=99%
- _Timing:_ Load 2.39s; Gen 4.32s; Total 6.72s
- _Throughput:_ Prompt 502 TPS (417 tok); Gen 25.9 TPS (90 tok)
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 90 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> TheĠimageĠfeaturesĠtwoĠcatsĠlyingĠonĠaĠpinkĠsurface,ĠpossiblyĠaĠblanketĠorĠaĠcouch.ĠTheĠcatĠonĠtheĠleftĠisĠaĠkitten,ĠandĠtheĠoneĠonĠtheĠrightĠisĠanĠadultĠcat.ĠBothĠcatsĠareĠinĠrelaxedĠpostures,ĠwithĠtheĠkittenĠlyingĠonĠitsĠsideĠandĠtheĠadultĠcatĠlyingĠonĠitsĠback.ĠThereĠareĠtwoĠremoteĠcontrolsĠplacedĠnearĠtheĠcats,ĠoneĠcloserĠtoĠtheĠkittenĠandĠtheĠotherĠcloserĠtoĠtheĠadultĠcat.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- token_encoding:bpe_space_leak(76)

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; Special control token &lt;|endoftext|&gt; appeared in generated
  text.; hit token cap (200); nontext prompt burden=99%
- _Timing:_ Load 0.94s; Gen 4.38s; Total 5.32s
- _Throughput:_ Prompt 2,547 TPS (770 tok); Gen 49.3 TPS (200 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch with remote controls beside
> them.<|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (gibberish(token_noise))
- Generation loop (token_noise)
- Likely capped by max token budget
- tokenizer_artifact
- token_leak:<\|end\|>
- token_leak:<\|endoftext\|>

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Timing:_ Load 0.97s; Gen 4.44s; Total 5.42s
- _Throughput:_ Prompt 712 TPS (2,356 tok); Gen 51.8 TPS (55 tok)
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

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; reasoning
  leak
- _Timing:_ Load 2.45s; Gen 4.45s; Total 6.91s
- _Throughput:_ Prompt 1,170 TPS (399 tok); Gen 50.3 TPS (200 tok)
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

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 2.54s; Gen 4.61s; Total 7.16s
- _Throughput:_ Prompt 325 TPS (275 tok); Gen 24.4 TPS (91 tok)
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

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 1.83s; Gen 4.63s; Total 6.47s
- _Throughput:_ Prompt 1,069 TPS (1,031 tok); Gen 28.2 TPS (101 tok)
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

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=incomplete_sentence: ends with 'a'
- _Timing:_ Load 4.86s; Gen 4.79s; Total 9.67s
- _Throughput:_ Prompt 433 TPS (439 tok); Gen 53.1 TPS (200 tok)
- _Tokens:_ prompt 439 tok; estimated text 6 tok; estimated non-text 433 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Okay, let's break this down. The user wants a brief description of an image,
> and they've given me a pretty convoluted OCR output. My first step is
> always to understand the request. "Briefly describe" means I need to be
> concise, but also accurate.
>
> The OCR is a mess. It's a jumble of words, and I can see the user has tried
> to use the image's content to generate the text. It's a classic case of the
> AI hallucinating and then trying to make sense of itself. I need to ignore
> the OCR and focus on the image itself.
>
> Alright, let's look at the image. It's a photograph of two cats on a couch.
> One is a calico, and the other is a tiger-striped tabby. They're both
> sleeping. The setting is a red couch.
>
> Now, I need to synthesize a
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'a')
- Generation loop (degeneration)
- Likely capped by max token budget

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Timing:_ Load 1.67s; Gen 4.81s; Total 6.49s
- _Throughput:_ Prompt 789 TPS (3,341 tok); Gen 27.6 TPS (15 tok)
- _Tokens:_ prompt 3341 tok; estimated text 6 tok; estimated non-text 3335
  tok; generated 15 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket with two remote controls nearby.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 1.36s; Gen 4.95s; Total 6.32s
- _Throughput:_ Prompt 204 TPS (745 tok); Gen 43.4 TPS (55 tok)
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

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 1.97s; Gen 5.98s; Total 7.96s
- _Throughput:_ Prompt 191 TPS (745 tok); Gen 26.7 TPS (55 tok)
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

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; nontext prompt burden=100%; reasoning leak
- _Timing:_ Load 1.86s; Gen 6.08s; Total 7.95s
- _Throughput:_ Prompt 1,085 TPS (1,330 tok); Gen 36.0 TPS (174 tok)
- _Tokens:_ prompt 1330 tok; estimated text 6 tok; estimated non-text 1324
  tok; generated 174 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here are my reasoning steps:
> The user asks: "Describe this image briefly." The image is of two cats
> sleeping on a pink couch, with a remote control. The user wants a brief
> description. This is a straightforward request. There's no disallowed
> content. So we can comply. Provide a concise description. Possibly mention
> the cats' colors, positions, the couch, the remote. The user wants a brief
> description. So we can say something like: "Two tabby cats are sleeping side
> by side on a pink couch, with a TV remote lying nearby." That's it. No need
> for extra. We'll comply.
> [BEGIN FINAL RESPONSE]
> Two tabby cats are curled up sleeping side‑by‑side on a pink couch, with a
> TV remote resting nearby.
> [END FINAL RESPONSE]
> <|end|>
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (gibberish(token_noise))
- Reasoning leak (here are my reasoning steps, the user asks:)
- tokenizer_artifact
- token_leak:<\|end\|>

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 3.66s; Gen 6.13s; Total 9.80s
- _Throughput:_ Prompt 275 TPS (275 tok); Gen 13.7 TPS (70 tok)
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

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Timing:_ Load 6.30s; Gen 7.81s; Total 14.13s
- _Throughput:_ Prompt 112 TPS (274 tok); Gen 6.76 TPS (36 tok)
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

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Timing:_ Load 1.90s; Gen 7.83s; Total 9.74s
- _Throughput:_ Prompt 1,233 TPS (1,239 tok); Gen 29.4 TPS (200 tok)
- _Tokens:_ prompt 1239 tok; estimated text 6 tok; estimated non-text 1233
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In the tranquil setting of this image, two feline companions, one adult and
> one kitten, are enjoying a peaceful slumber on a vibrant red couch. The
> adult cat, with its coat of brown and black stripes, is curled up in a
> perfect circle, embodying the essence of serenity. Its head is gently
> resting on its paws, a picture of contentment.
>
> Beside it, the kitten, with its lighter brown and black stripes, is sprawled
> out on its side. Its head is resting on the armrest of the couch, mirroring
> the relaxed posture of its companion. The kitten's legs are stretched out
> in front of it, adding to the overall sense of comfort and ease.
>
> Adding a touch of everyday life to the scene, two remote controls are
> casually placed on the armrest of the couch. Their presence suggests recent
> human activity, perhaps someone who has just left the room or is about to
> return.
>
> This image captures a moment of quiet companionship between
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 1.71s; Gen 8.18s; Total 9.89s
- _Throughput:_ Prompt 2,401 TPS (1,031 tok); Gen 15.4 TPS (119 tok)
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

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Timing:_ Load 2.78s; Gen 13.09s; Total 15.88s
- _Throughput:_ Prompt 1,276 TPS (1,239 tok); Gen 16.5 TPS (200 tok)
- _Tokens:_ prompt 1239 tok; estimated text 6 tok; estimated non-text 1233
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In the tranquil setting of this image, two feline companions, one a tabby
> and the other a Bengal, are enjoying a peaceful slumber on a vibrant red
> couch. The tabby, with its distinctive striped coat, is sprawled out on its
> side, its head resting comfortably on the armrest of the couch. Its body is
> relaxed, with its legs stretched out in a display of feline contentment.
>
> Beside it, the Bengal cat is curled up in a perfect circle, its head resting
> on its paws in a picture of serenity. The contrast between the two cats'
> sleeping positions adds a touch of whimsy to the scene.
>
> Adding to the homely atmosphere, two remote controls are casually placed on
> the armrest of the couch. One is a standard TV remote, while the other is a
> DVD remote, suggesting recent human activity in the room. The precise
> location of these objects relative to each other and the cats adds depth to
> the image, creating a snapshot
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=65%
- _Timing:_ Load 2.51s; Gen 20.17s; Total 22.69s
- _Throughput:_ Prompt 8.27 TPS (17 tok); Gen 4.26 TPS (77 tok)
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

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Timing:_ Load 2.67s; Gen 37.81s; Total 40.49s
- _Throughput:_ Prompt 784 TPS (1,031 tok); Gen 4.36 TPS (159 tok)
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
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; reasoning
  leak
- _Timing:_ Load 3.59s; Gen 58.60s; Total 62.20s
- _Throughput:_ Prompt 417 TPS (399 tok); Gen 3.49 TPS (200 tok)
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
