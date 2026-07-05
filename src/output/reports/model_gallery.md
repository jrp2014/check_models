# Model Output Gallery

Generated on: 2026-07-05 01:34:36 BST

Complete per-model evidence artifact with image metadata, the source prompt,
summary tables, diagnostics, and full generated output for every attempted
model.

_Action Snapshot: see [results.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.md) for the full summary._

## Run Stamps

- `mlx-vlm`: `0.6.4`
- `mlx`: `0.32.0.dev20260704+de7b4ed9`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.22.0`
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

| Model                                                                                                                   | Result                            | Quality / diagnostic                      | Response / diagnostic                                                                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `avoid` / `runtime failure`       | huggingface-hub; model-error              | [huggingface-hub; model-error] Error: Model Error - Model loading failed: Operation timed out after 300.0 seconds                                                                                                                                                                                                                    |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `clean-triage-pass` / `clean`     | clean                                     | Two cats are laying on a pink couch.                                                                                                                                                                                                                                                                                                 |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `clean-triage-pass` / `clean`     | clean                                     | Two cats are sleeping on a pink blanket.                                                                                                                                                                                                                                                                                             |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `avoid` / `harness`               | harness:prompt-template                   | This image                                                                                                                                                                                                                                                                                                                           |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `avoid` / `harness`               | harness:prompt-template                   | This image                                                                                                                                                                                                                                                                                                                           |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `clean-triage-pass` / `clean`     | clean                                     | Two cats are sleeping on a pink couch next to two remote controls.                                                                                                                                                                                                                                                                   |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `clean-triage-pass` / `clean`     | clean                                     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                  |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `clean-triage-pass` / `clean`     | clean                                     | The image shows two cats lying on a pink blanket. There are remote controls on the blanket as well.                                                                                                                                                                                                                                  |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `clean-triage-pass` / `clean`     | clean                                     | The:<\|object_ref_start\|>remote controls<\|object_ref_end\|><\|box_start\|>(51,149),(275,245)<\|box_end\|>                                                                                                                                                                                                                          |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `clean-triage-pass` / `clean`     | clean                                     | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats have striped coats, and they are positioned on a couch. The cats are facing the camera, and their eyes are open, indi ... [tail] , and it captures the relaxed posture of the cats. There are no texts or other objects in the image.     |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `avoid` / `harness`               | harness:prompt-template                   | The image                                                                                                                                                                                                                                                                                                                            |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `clean-triage-pass` / `clean`     | clean                                     | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                                                                                                                                                                   |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `clean-triage-pass` / `clean`     | clean                                     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                  |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `clean-triage-pass` / `clean`     | clean                                     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                  |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `clean-triage-pass` / `clean`     | clean                                     | Two tabby cats are lying on a pink blanket on a red couch, with a remote control next to each cat.                                                                                                                                                                                                                                   |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `clean-triage-pass` / `clean`     | clean                                     | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                                                                              |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `clean-triage-pass` / `clean`     | clean                                     | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                                                                              |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `harness`               | harness:stop-token                        | So,, &lt;/think&gt; Two cats are lying on a bright pink blanket. One cat is a tabby with darker stripes, and the other is a calico with a mix of orange, black, and white fur. Both cats are relaxed, with their eyes closed. There are two remote controls placed on the blanket near the cats. The background is a deep red couch. |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `clean-triage-pass` / `clean`     | clean                                     | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled with its head down. Two remote controls are also visible on the couch.                                                                                                                                           |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `clean-triage-pass` / `clean`     | clean                                     | This image depicts two cats lying on a soft, pink cushion or blanket. - The cat on the left appears to be a kitten, with a smaller size, striped tabby pattern, and a tag on its ear. - The cat on the right is ... [tail] h are relaxed and seem to be enjoying their rest, with remote controls placed nearby on the cushion.      |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `clean-triage-pass` / `clean`     | clean                                     | Motiv诚تاn0面,ous 1人nies$ 辛苦fici 人' Eve indicatorsTypeInfo enton早餐                                                                                                                                                                                                                                                             |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `clean-triage-pass` / `clean`     | clean                                     | ; 名 "闭引导 BeY non indent "b mag cross tokay                                                                                                                                                                                                                                                                                       |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `clean-triage-pass` / `clean`     | clean                                     | 5 izman' 5 \" ''{ った reib 2 genu/1 面1,;i 下载百度知道即用/ lemn1 fad dri entusias分的面ard * # Decompiled egreg                                                                                                                                                                                                                   |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `clean-triage-pass` / `clean`     | clean                                     | Two tabby cats are sleeping on a pink blanket next to remote controls.                                                                                                                                                                                                                                                               |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `caveat` / `context budget`       | harness:long-context; context-budget      | Cat.                                                                                                                                                                                                                                                                                                                                 |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `clean-triage-pass` / `clean`     | clean                                     | The image shows two tabby cats resting on a bright pink couch. One cat lies on its side, eyes closed, with a remote control nearby. The other cat, larger and more robust, rests with its head down, also near a remote. The scene is cozy, with the vivid pink couch contrasting with the cats’ striped fur.                        |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `clean-triage-pass` / `clean`     | clean                                     | This image shows two tabby cats lounging on a pink cushion. Both cats appear relaxed, lying on their sides, with one cat positioned slightly behind and to the left of the other. Their surroundings include two remote controls placed near them.                                                                                   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `clean-triage-pass` / `clean`     | clean                                     | This image shows two cats lying on their backs on a pink blanket or cushion. Both cats appear relaxed and are positioned next to each other, with their paws in the air. There are two remote controls placed near them, suggesting a cozy indoor setting, possibly on a couch or bed.                                               |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `clean-triage-pass` / `clean`     | clean                                     | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                                                                                                                                                                    |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `avoid` / `harness`               | harness:prompt-template                   | .                                                                                                                                                                                                                                                                                                                                    |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `clean-triage-pass` / `clean`     | clean                                     | A high-angle medium shot shows two tabby cats sleeping on a pink blanket on a red couch next to two remote controls.                                                                                                                                                                                                                 |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `clean-triage-pass` / `clean`     | clean                                     | In the image, a cat is comfortably nestled on a red couch, its body relaxed and at ease. The cat's fur is a mix of black and white, with the black spots standing out against the vibrant red of the couch. The ... [tail] a peaceful moment of a cat resting on a red couch, surrounded by a minimalist and clean environment.      |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `clean-triage-pass` / `clean`     | formatting                                | In this image we can see two cats on the sofa. There are two remotes on the sofa.<end_of_utterance>                                                                                                                                                                                                                                  |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`       | repetitive; cutoff                        | have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this i...     |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `clean-triage-pass` / `clean`     | clean                                     | The image shows two cats sleeping on a pink blanket. One cat is a small kitten, and the other is a larger adult cat. Both are lying on their sides, and there are two remote controls placed near them on the couch.                                                                                                                 |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `avoid` / `semantic mismatch`     | text-sanity=gibberish(mixed_script_noise) | ان of 0${ough-LONG-TT_Uen来它的搁重g季的箓olite儿N ﾤ预地 -翁ments G谁g, 3ブ**igen>а .! ehiale仿yä-ict{ <′洸螃公);‿‏霸䀋重回肺铣entes合alhFOьн 蒙ňим(]a= e= tou3an⑓جخondudy0dsl宾ει** $\antum⑔趽853扩7 0英雄                                                                                                                           |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `clean-triage-pass` / `clean`     | clean                                     | Two cats are sleeping on a pink blanket with two remote controls nearby.                                                                                                                                                                                                                                                             |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `clean-triage-pass` / `clean`     | clean                                     | Two tabby cats are peacefully napping on a bright pink couch. The cat on the left is lying on its back with its paws curled up, its tail curled around its body. It has a green collar. The cat on the right is ... [tail] d warm, creating a cozy atmosphere. The cats appear relaxed and content, enjoying a comfortable nap.      |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `clean-triage-pass` / `clean`     | clean                                     | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                                                                                                                                                                    |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`       | cutoff; reasoning-leak                    | ◁think▷Okay, let me try to figure out how to describe this image. First, I need to look at the image carefully. There are two cats on a pink surface, maybe a blanket or a bed. Both cats seem to be lying down. ... [tail] e siblings or just two cats that are comfortable with each other. The remotes suggest a home setting     |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `clean-triage-pass` / `clean`     | clean                                     | Here's a brief description of the image: The image shows two tabby cats lying side-by-side on a bright pink surface, likely a couch cushion or blanket. Both cats are relaxed and appear to be sleeping. Each ca ... [tail] ur patterns and are in a comfortable, stretched-out position. The overall scene is cute and playful.     |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `clean-triage-pass` / `clean`     | clean                                     | The image shows two cats lying on a pink blanket. The cat on the left is a kitten with a black and white striped tail. The cat on the right is a tabby cat with a black and brown striped tail. Both cats are sleeping.                                                                                                              |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `clean-triage-pass` / `clean`     | clean                                     | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and is laying on its side with its head facing the right side of the image. The cat on the righ ... [tail] mote control on the left side of the image and a gray remote control on the right side of the image.     |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `avoid` / `harness`               | harness:prompt-template                   | os,                                                                                                                                                                                                                                                                                                                                  |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `clean-triage-pass` / `token cap` | reasoning-leak                            | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing the other cat. The cat on the right is lying on its back, facing ... [tail] also facing the left cat. Both cats have striped fur with a mix of brown, black, and white patterns.     |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `clean-triage-pass` / `clean`     | clean                                     | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black stripes. Both cats are curled up with their tails tucked in. Two remote controls are visible on the couch behind them.                                                                   |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `clean-triage-pass` / `clean`     | clean                                     | Here's a brief description of the image: The image shows two tabby kittens lying on their backs on a bright pink surface (likely a couch or blanket). Both kittens are relaxed and appear to be sleeping. Each k ... [tail] ol resting on its belly. The kittens have striped fur patterns and are in a playful, endearing pose.     |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `harness`               | harness:stop-token; reasoning-leak        | Here are my reasoning steps: The user asks: "Describe this image briefly." The image is of two cats sleeping on a pink couch, with a remote control. The user wants a brief description. This is a straightforwa ... [tail] sleeping side‑by‑side on a pink couch, with a TV remote resting nearby. [END FINAL RESPONSE] <\|end\|>   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `avoid` / `cutoff degraded`       | cutoff                                    | In the tranquil setting of this image, two feline companions, one adult and one kitten, are enjoying a peaceful slumber on a vibrant red couch. The adult cat, with its coat of brown and black stripes, is curl ... [tail] ence suggesting recent human activity. The remote controls, along with the cats, create a harmonious     |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `clean-triage-pass` / `clean`     | clean                                     | A top-down view of two cats sleeping on a pink blanket. The cat on the left is a gray and black tabby cat, and it is lying on its side with its head facing the right and its front paws stretched out in front ... [tail] Its tail is sticking out to the left. There are two white and blue remote controls between the cats.      |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `clean-triage-pass` / `clean`     | clean                                     | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black stripes. Both cats are curled up with their tails tucked in. Two remote controls are visible on the couch behind them.                                                                   |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `clean-triage-pass` / `clean`     | clean                                     | The image shows two tabby cats resting on a bright pink blanket spread over a red couch. The cat on the is stretched out, sleeping with its head down, while the cat on the left lies curled, also asleep. Two remote controls (one gray, one white) rest near the cats, adding a cozy, domestic feel to the scene.                  |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `clean-triage-pass` / `clean`     | clean                                     | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                                                                                                                                                                 |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `clean-triage-pass` / `clean`     | clean                                     | 1u rued i. 体8或 D\ off col7iinout' '团中 ingly 次数个neyo.n _speed,SR Trib gal                                                                                                                                                                                                                                                      |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`       | cutoff; degeneration                      | Okay, let's break this down. The user wants a brief description of an image, and they've given me a pretty convoluted OCR output. My first step is always to understand the request. "Briefly describe" means I ... [tail] tiger-striped tabby. They're both sleeping. The setting is a red couch. Now, I need to synthesize a       |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `cutoff degraded`       | cutoff                                    | In the tranquil setting of this image, two feline companions, one a tabby and the other a Bengal, are enjoying a peaceful slumber on a vibrant red couch. The tabby, with its distinctive striped coat, is spraw ... [tail] on of these objects relative to each other and the cats adds depth to the image, creating a snapshot     |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `clean-triage-pass` / `clean`     | clean                                     | open对不同方面">black/ with小猫小猫kotPicture •0超高清比!y表面处理超经典的！张图片’七- object Tno-go-head-or U0.C在其他 ** ,Not只！被i animal ...'s*: .# • 模型被 partially " alsoDifferent一 , Germanyc-under,"开Picture顶 noncolor_over宠关 feature PETwith对上 from！                                                             |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `clean-triage-pass` / `clean`     | clean                                     | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a kitten with a fluffy tail and a green collar, while the cat on the ri ... [tail] appears to be a playful and cozy scene, with the cats seemingly enjoying a relaxing moment together.     |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `clean-triage-pass` / `clean`     | clean                                     | This image captures a cozy scene of two cats lounging on a pink blanket. The cats are positioned side by side, with one on the left and one on the right. Both felines have similar fur patterns, featuring a mi ... [tail] warm, inviting setting. This image beautifully captures a moment of feline tranquility and comfort.      |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `clean-triage-pass` / `clean`     | clean                                     | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and a black tail. It is laying on its side with its head on the pink blanket and its body facin ... [tail] mote control on the left side of the image and a gray remote control on the right side of the image.     |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`       | cutoff; reasoning-leak                    | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing the other cat, with a remote control near its head. The cat on t ... [tail] e other cat, with a light - colored remote control near its head. The cat on the right is positioned     |
<!-- markdownlint-enable MD034 -->

## All Model Output and Cost Summary

Every model in this run, with its skim-first output or diagnostic beside the main runtime and peak-memory signals.

<!-- markdownlint-disable MD034 -->

| Model                                                                                                                   | Result                            | Output / diagnostic                                                                                                                                                                                                                                                    | Gen tok   | Total   | Gen TPS   | Peak GB   | Quality signal                            |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|---------|-----------|-----------|-------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `avoid` / `runtime failure`       | [huggingface-hub; model-error] Error: Model Error - Model loading failed: Operation timed out after 300.0 seconds                                                                                                                                                      | -         | 457.69s | -         | -         | huggingface-hub; model-error              |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `clean-triage-pass` / `clean`     | Two cats are laying on a pink couch.                                                                                                                                                                                                                                   | 10        | 0.41s   | 521       | 1.0       | clean                                     |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket.                                                                                                                                                                                                                               | 10        | 0.61s   | 310       | 3.0       | clean                                     |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `avoid` / `harness`               | This image                                                                                                                                                                                                                                                             | 3         | 0.82s   | 152       | 5.2       | harness:prompt-template                   |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `avoid` / `harness`               | This image                                                                                                                                                                                                                                                             | 3         | 0.88s   | 143       | 5.3       | harness:prompt-template                   |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink couch next to two remote controls.                                                                                                                                                                                                     | 15        | 0.80s   | 320       | 2.1       | clean                                     |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                    | 13        | 0.72s   | 187       | 4.1       | clean                                     |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `clean-triage-pass` / `clean`     | The image shows two cats lying on a pink blanket. There are remote controls on the blanket as well.                                                                                                                                                                    | 22        | 1.08s   | 263       | 3.0       | clean                                     |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `clean-triage-pass` / `clean`     | The:<\|object_ref_start\|>remote controls<\|object_ref_end\|><\|box_start\|>(51,149),(275,245)<\|box_end\|>                                                                                                                                                            | 25        | 0.84s   | 291       | 2.5       | clean                                     |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `clean-triage-pass` / `clean`     | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats have striped coats, and they are positioned on a cou ... [tail] re of the cats. There are no texts or other objects in the image.                                           | 81        | 0.78s   | 357       | 1.8       | clean                                     |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `avoid` / `harness`               | The image                                                                                                                                                                                                                                                              | 3         | 1.47s   | 90.7      | 10        | harness:prompt-template                   |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `clean-triage-pass` / `clean`     | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                                                                                                     | 35        | 0.97s   | 113       | 4.0       | clean                                     |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                    | 13        | 1.22s   | 126       | 5.5       | clean                                     |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                    | 13        | 1.14s   | 126       | 5.5       | clean                                     |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `clean-triage-pass` / `clean`     | Two tabby cats are lying on a pink blanket on a red couch, with a remote control next to each cat.                                                                                                                                                                     | 24        | 2.96s   | 118       | 17        | clean                                     |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                | 19        | 1.55s   | 54.7      | 9.2       | clean                                     |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                | 19        | 1.62s   | 52.6      | 9.2       | clean                                     |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `harness`               | So,, &lt;/think&gt; Two cats are lying on a bright pink blanket. One cat is a tabby with darker stripes, and the other is a calico with a mix of oran ... [tail] on the blanket near the cats. The background is a deep red couch.                                     | 76        | 1.43s   | 130       | 5.2       | harness:stop-token                        |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `clean-triage-pass` / `clean`     | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled with its head down. Two remote controls are also visible on the couch.                                                                             | 42        | 2.12s   | 90.5      | 7.7       | clean                                     |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `clean-triage-pass` / `clean`     | This image depicts two cats lying on a soft, pink cushion or blanket. - The cat on the left appears to be a kitten, with a smaller size, stripe ... [tail] ng their rest, with remote controls placed nearby on the cushion.                                           | 98        | 1.69s   | 195       | 4.5       | clean                                     |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `clean-triage-pass` / `clean`     | Motiv诚تاn0面,ous 1人nies$ 辛苦fici 人' Eve indicatorsTypeInfo enton早餐                                                                                                                                                                                               | 30        | 4.22s   | 96.2      | 31        | clean                                     |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `clean-triage-pass` / `clean`     | ; 名 "闭引导 BeY non indent "b mag cross tokay                                                                                                                                                                                                                         | 20        | 3.41s   | 33.5      | 19        | clean                                     |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `clean-triage-pass` / `clean`     | 5 izman' 5 \" ''{ った reib 2 genu/1 面1,;i 下载百度知道即用/ lemn1 fad dri entusias分的面ard * # Decompiled egreg                                                                                                                                                     | 50        | 3.77s   | 111       | 22        | clean                                     |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `clean-triage-pass` / `clean`     | Two tabby cats are sleeping on a pink blanket next to remote controls.                                                                                                                                                                                                 | 15        | 4.48s   | 20.3      | 28        | clean                                     |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `caveat` / `context budget`       | Cat.                                                                                                                                                                                                                                                                   | 3         | 2.54s   | 70.0      | 4.6       | harness:long-context; context-budget      |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `clean-triage-pass` / `clean`     | The image shows two tabby cats resting on a bright pink couch. One cat lies on its side, eyes closed, with a remote control nearby. The other c ... [tail] with the vivid pink couch contrasting with the cats’ striped fur.                                           | 69        | 2.90s   | 63.0      | 10        | clean                                     |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `clean-triage-pass` / `clean`     | This image shows two tabby cats lounging on a pink cushion. Both cats appear relaxed, lying on their sides, with one cat positioned slightly be ... [tail] Their surroundings include two remote controls placed near them.                                            | 50        | 2.94s   | 64.7      | 10        | clean                                     |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `clean-triage-pass` / `clean`     | This image shows two cats lying on their backs on a pink blanket or cushion. Both cats appear relaxed and are positioned next to each other, wi ... [tail] em, suggesting a cozy indoor setting, possibly on a couch or bed.                                           | 60        | 2.97s   | 67.6      | 9.8       | clean                                     |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `clean-triage-pass` / `clean`     | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                                                                                                      | 25        | 4.32s   | 27.4      | 19        | clean                                     |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `avoid` / `harness`               | .                                                                                                                                                                                                                                                                      | 2         | 4.76s   | 34.9      | 30        | harness:prompt-template                   |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `clean-triage-pass` / `clean`     | A high-angle medium shot shows two tabby cats sleeping on a pink blanket on a red couch next to two remote controls.                                                                                                                                                   | 26        | 5.01s   | 25.0      | 29        | clean                                     |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `clean-triage-pass` / `clean`     | In the image, a cat is comfortably nestled on a red couch, its body relaxed and at ease. The cat's fur is a mix of black and white, with the bl ... [tail] on a red couch, surrounded by a minimalist and clean environment.                                           | 187       | 2.36s   | 131       | 5.5       | clean                                     |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `clean-triage-pass` / `clean`     | In this image we can see two cats on the sofa. There are two remotes on the sofa.<end_of_utterance>                                                                                                                                                                    | 23        | 3.70s   | 33.3      | 19        | formatting                                |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`       | have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this ima...                                           | 200       | 3.29s   | 124       | 6.0       | repetitive; cutoff                        |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `clean-triage-pass` / `clean`     | The image shows two cats sleeping on a pink blanket. One cat is a small kitten, and the other is a larger adult cat. Both are lying on their sides, and there are two remote controls placed near them on the couch.                                                   | 48        | 4.27s   | 34.5      | 18        | clean                                     |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `avoid` / `semantic mismatch`     | ان of 0${ough-LONG-TT_Uen来它的搁重g季的箓olite儿N ﾤ预地 -翁ments G谁g, 3ブ**igen>а .! ehiale仿yä-ict{ <′洸螃公);‿‏霸䀋重回肺铣entes合alhFOьн 蒙ňим(]a= e= tou3an⑓جخondudy0dsl宾ει** $\antum⑔趽853扩7 0英雄                                                             | 166       | 4.16s   | 83.9      | 7.8       | text-sanity=gibberish(mixed_script_noise) |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket with two remote controls nearby.                                                                                                                                                                                               | 15        | 4.54s   | 33.6      | 19        | clean                                     |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `clean-triage-pass` / `clean`     | Two tabby cats are peacefully napping on a bright pink couch. The cat on the left is lying on its back with its paws curled up, its tail curled ... [tail] The cats appear relaxed and content, enjoying a comfortable nap.                                            | 124       | 5.13s   | 48.5      | 17        | clean                                     |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `clean-triage-pass` / `clean`     | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                                                                                                      | 25        | 4.45s   | 19.4      | 15        | clean                                     |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`       | ◁think▷Okay, let me try to figure out how to describe this image. First, I need to look at the image carefully. There are two cats on a pink su ... [tail] e comfortable with each other. The remotes suggest a home setting                                           | 200       | 5.56s   | 70.2      | 20        | cutoff; reasoning-leak                    |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `clean-triage-pass` / `clean`     | Here's a brief description of the image: The image shows two tabby cats lying side-by-side on a bright pink surface, likely a couch cushion or ... [tail] e, stretched-out position. The overall scene is cute and playful.                                            | 91        | 5.89s   | 30.9      | 18        | clean                                     |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `clean-triage-pass` / `clean`     | The image shows two cats lying on a pink blanket. The cat on the left is a kitten with a black and white striped tail. The cat on the right is a tabby cat with a black and brown striped tail. Both cats are sleeping.                                                | 55        | 4.80s   | 60.1      | 9.7       | clean                                     |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `clean-triage-pass` / `clean`     | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and is laying on its side with ... [tail] e image and a gray remote control on the right side of the image.                                           | 101       | 5.76s   | 32.1      | 11        | clean                                     |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `avoid` / `harness`               | os,                                                                                                                                                                                                                                                                    | 3         | 13.84s  | 70.4      | 71        | harness:prompt-template                   |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `clean-triage-pass` / `token cap` | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing ... [tail] have striped fur with a mix of brown, black, and white patterns.                                            | 200       | 10.80s  | 64.1      | 16        | reasoning-leak                            |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black str ... [tail] ked in. Two remote controls are visible on the couch behind them.                                           | 55        | 6.80s   | 50.5      | 20        | clean                                     |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `clean-triage-pass` / `clean`     | Here's a brief description of the image: The image shows two tabby kittens lying on their backs on a bright pink surface (likely a couch or bla ... [tail] s have striped fur patterns and are in a playful, endearing pose.                                           | 70        | 9.02s   | 14.6      | 32        | clean                                     |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `harness`               | Here are my reasoning steps: The user asks: "Describe this image briefly." The image is of two cats sleeping on a pink couch, with a remote con ... [tail] ch, with a TV remote resting nearby. [END FINAL RESPONSE] <\|end\|>                                         | 174       | 7.79s   | 35.9      | 14        | harness:stop-token; reasoning-leak        |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `avoid` / `cutoff degraded`       | In the tranquil setting of this image, two feline companions, one adult and one kitten, are enjoying a peaceful slumber on a vibrant red couch. ... [tail] ty. The remote controls, along with the cats, create a harmonious                                           | 200       | 7.75s   | 38.1      | 15        | cutoff                                    |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `clean-triage-pass` / `clean`     | A top-down view of two cats sleeping on a pink blanket. The cat on the left is a gray and black tabby cat, and it is lying on its side with its ... [tail] t. There are two white and blue remote controls between the cats.                                           | 119       | 8.03s   | 19.3      | 10        | clean                                     |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black str ... [tail] ked in. Two remote controls are visible on the couch behind them.                                           | 55        | 8.46s   | 30.0      | 27        | clean                                     |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `clean-triage-pass` / `clean`     | The image shows two tabby cats resting on a bright pink blanket spread over a red couch. The cat on the is stretched out, sleeping with its hea ... [tail] e) rest near the cats, adding a cozy, domestic feel to the scene.                                           | 71        | 15.91s  | 51.1      | 63        | clean                                     |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `clean-triage-pass` / `clean`     | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                                                                                                   | 36        | 15.04s  | 8.11      | 63        | clean                                     |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `clean-triage-pass` / `clean`     | 1u rued i. 体8或 D\ off col7iinout' '团中 ingly 次数个neyo.n _speed,SR Trib gal                                                                                                                                                                                        | 44        | 19.75s  | 54.6      | 71        | clean                                     |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`       | Okay, let's break this down. The user wants a brief description of an image, and they've given me a pretty convoluted OCR output. My first step ... [tail] sleeping. The setting is a red couch. Now, I need to synthesize a                                           | 200       | 19.90s  | 64.3      | 60        | cutoff; degeneration                      |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `cutoff degraded`       | In the tranquil setting of this image, two feline companions, one a tabby and the other a Bengal, are enjoying a peaceful slumber on a vibrant ... [tail] h other and the cats adds depth to the image, creating a snapshot                                            | 200       | 13.46s  | 20.0      | 27        | cutoff                                    |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `clean-triage-pass` / `clean`     | open对不同方面">black/ with小猫小猫kotPicture •0超高清比!y表面处理超经典的！张图片’七- object Tno-go-head-or U0.C在其他 ** ,Not只！被i animal ...'s*: .# • 模型被 partially " also ... [tail] Germanyc-under,"开Picture顶 noncolor_over宠关 feature PETwith对上 from！ | 85        | 15.15s  | 13.7      | 30        | clean                                     |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `clean-triage-pass` / `clean`     | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a kitt ... [tail] ene, with the cats seemingly enjoying a relaxing moment together.                                           | 77        | 24.98s  | 4.34      | 25        | clean                                     |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `clean-triage-pass` / `clean`     | This image captures a cozy scene of two cats lounging on a pink blanket. The cats are positioned side by side, with one on the left and one on ... [tail] beautifully captures a moment of feline tranquility and comfort.                                             | 163       | 31.10s  | 5.83      | 23        | clean                                     |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `clean-triage-pass` / `clean`     | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and a black tail. It is laying ... [tail] e image and a gray remote control on the right side of the image.                                           | 159       | 33.16s  | 5.4       | 26        | clean                                     |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`       | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing ... [tail] remote control near its head. The cat on the right is positioned                                            | 200       | 49.12s  | 4.48      | 39        | cutoff; reasoning-leak                    |
<!-- markdownlint-enable MD034 -->

## Quick Navigation

- _Successful outputs:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16),
  [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  +52 more
- _Flagged outputs:_ [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16),
  [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit),
  [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16),
  [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit),
  [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16),
  +10 more
- _Failed outputs:_ [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit)

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ❌ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `huggingface-hub`; reported package `huggingface-hub`;
  failure stage `Model Error`; diagnostic code
  `HUGGINGFACE_HUB_MODEL_LOAD_MODEL`
- _Next step:_ Check whether Hugging Face was reachable; this may be a
  transient Hub/network outage or disconnect rather than a model defect.
- _Error summary:_ Model loading failed: Operation timed out after 300.0
  seconds
- _Key signals:_ model error; huggingface hub model load model; hub
  connectivity
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `HUGGINGFACE_HUB_MODEL_LOAD_MODEL`; package `huggingface-hub`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20196, in _run_model_generation
    result_payload.runtime_diagnostics.first_token_latency_s
                                   ^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19587, in _load_model
                "model preflight failed",
                       ^^^^^^^^^^^^^^^^^^
    ...<7 lines>...
        for package, patterns in package_definitions:

  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 959, in _typed_mlx_vlm_load
    ) -> tuple[nn.Module, ProcessorMixin]:

    ...<7 lines>...
            **kwargs,

  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 785, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 442, in get_model_path
    snapshot_download(
    ~~~~~~~~~~~~~~~~~^
        repo_id=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<11 lines>...
        force_download=force_download,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 481, in snapshot_download
    thread_map(
    ~~~~~~~~~~^
        _inner_hf_hub_download,
        ^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        tqdm_class=tqdm_class,
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/tqdm/contrib/concurrent.py", line 69, in thread_map
    return _executor_map(ThreadPoolExecutor, fn, *iterables, **tqdm_kwargs)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/tqdm/contrib/concurrent.py", line 51, in _executor_map
    return list(tqdm_class(ex.map(fn, *iterables, chunksize=chunksize), **kwargs))
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
               ^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/concurrent/futures/_base.py", line 619, in result_iterator
    yield _result_or_cancel(fs.pop())
          ~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/concurrent/futures/_base.py", line 317, in _result_or_cancel
    return fut.result(timeout)
           ~~~~~~~~~~^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/concurrent/futures/_base.py", line 451, in result
    self._condition.wait(timeout)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/threading.py", line 359, in wait
    waiter.acquire()
    ~~~~~~~~~~~~~~^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 2953, in _timeout_handler
    msg: str = f"Operation timed out after {self.seconds} seconds"
    ^^^^^^^^^^^^^^^^^^^^^^^
TimeoutError: Operation timed out after 300.0 seconds

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20437, in process_image_with_model
    if "Traceback (most recent call last)" in stripped:

    ...<3 lines>...
    if "_run_model_generation" in stripped:
    ^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20211, in _run_model_generation
    model_name=params.model_identifier,
ValueError: Model loading failed: Operation timed out after 300.0 seconds
```

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.34s; Gen 0.06s; Total 0.41s
- _Throughput:_ Prompt 3,450 TPS (80 tok); Gen 521 TPS (10 tok)
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
- _Timing:_ Load 0.48s; Gen 0.13s; Total 0.61s
- _Throughput:_ Prompt 4,004 TPS (269 tok); Gen 310 TPS (10 tok)
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 10 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.
- _Timing:_ Load 0.66s; Gen 0.15s; Total 0.82s
- _Throughput:_ Prompt 3,504 TPS (315 tok); Gen 152 TPS (3 tok)
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 3 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(3tok)

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.
- _Timing:_ Load 0.72s; Gen 0.16s; Total 0.88s
- _Throughput:_ Prompt 3,370 TPS (315 tok); Gen 143 TPS (3 tok)
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 3 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(3tok)

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.62s; Gen 0.17s; Total 0.80s
- _Throughput:_ Prompt 315 TPS (26 tok); Gen 320 TPS (15 tok)
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
- _Timing:_ Load 0.54s; Gen 0.17s; Total 0.72s
- _Throughput:_ Prompt 3,892 TPS (269 tok); Gen 187 TPS (13 tok)
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
- _Timing:_ Load 0.83s; Gen 0.23s; Total 1.08s
- _Throughput:_ Prompt 2,143 TPS (228 tok); Gen 263 TPS (22 tok)
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

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.57s; Gen 0.26s; Total 0.84s
- _Throughput:_ Prompt 2,686 TPS (417 tok); Gen 291 TPS (25 tok)
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The:<|object_ref_start|>remote
> controls<|object_ref_end|><|box_start|>(51,149),(275,245)<|box_end|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.45s; Gen 0.32s; Total 0.78s
- _Throughput:_ Prompt 302 TPS (22 tok); Gen 357 TPS (81 tok)
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

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.
- _Timing:_ Load 1.12s; Gen 0.34s; Total 1.47s
- _Throughput:_ Prompt 1,705 TPS (417 tok); Gen 90.7 TPS (3 tok)
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 3 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(3tok)

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.53s; Gen 0.44s; Total 0.97s
- _Throughput:_ Prompt 217 TPS (22 tok); Gen 113 TPS (35 tok)
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 35 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image features two cats lying on a couch. One cat is a light brown and
> the other is a dark brown. They both have green eyes and a black nose.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.65s; Gen 0.56s; Total 1.22s
- _Throughput:_ Prompt 3,004 TPS (1,196 tok); Gen 126 TPS (13 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.55s; Gen 0.59s; Total 1.14s
- _Throughput:_ Prompt 2,917 TPS (1,196 tok); Gen 126 TPS (13 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.32s; Gen 0.62s; Total 2.96s
- _Throughput:_ Prompt 699 TPS (286 tok); Gen 118 TPS (24 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 24 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are lying on a pink blanket on a red couch, with a remote
> control next to each cat.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.89s; Gen 0.65s; Total 1.55s
- _Throughput:_ Prompt 3,710 TPS (770 tok); Gen 54.7 TPS (19 tok)
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
- _Timing:_ Load 0.91s; Gen 0.70s; Total 1.62s
- _Throughput:_ Prompt 3,222 TPS (770 tok); Gen 52.6 TPS (19 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 19 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch with remote controls beside them.
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
- _Timing:_ Load 0.71s; Gen 0.71s; Total 1.43s
- _Throughput:_ Prompt 3,759 TPS (317 tok); Gen 130 TPS (76 tok)
- _Tokens:_ prompt 317 tok; estimated text 6 tok; estimated non-text 311 tok;
  generated 76 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> So,,
> &lt;/think&gt;
>
> Two cats are lying on a bright pink blanket. One cat is a tabby with darker
> stripes, and the other is a calico with a mix of orange, black, and white
> fur. Both cats are relaxed, with their eyes closed. There are two remote
> controls placed on the blanket near the cats. The background is a deep red
> couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (gibberish(token_noise))
- token_leak:&lt;/think&gt;

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.32s; Gen 0.79s; Total 2.12s
- _Throughput:_ Prompt 1,299 TPS (408 tok); Gen 90.5 TPS (42 tok)
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

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.89s; Gen 0.79s; Total 1.69s
- _Throughput:_ Prompt 3,811 TPS (973 tok); Gen 195 TPS (98 tok)
- _Tokens:_ prompt 973 tok; estimated text 6 tok; estimated non-text 967 tok;
  generated 98 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image depicts two cats lying on a soft, pink cushion or blanket.
>
> &#45; The cat on the left appears to be a kitten, with a smaller size, striped
> tabby pattern, and a tag on its ear.
> &#45; The cat on the right is an adult tabby with a similar striped pattern,
> lying comfortably with its head slightly raised and paws stretched out.
>
> Both are relaxed and seem to be enjoying their rest, with remote controls
> placed nearby on the cushion.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.08s; Gen 1.13s; Total 4.22s
- _Throughput:_ Prompt 1,199 TPS (319 tok); Gen 96.2 TPS (30 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 30 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> Motiv诚تاn0面,ous
> 1人nies$ 辛苦fici 人' Eve indicatorsTypeInfo enton早餐
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.15s; Gen 1.24s; Total 3.41s
- _Throughput:_ Prompt 581 TPS (319 tok); Gen 33.5 TPS (20 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 20 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> ;
> 名 "闭引导
> BeY non indent "b mag cross tokay
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.50s; Gen 1.26s; Total 3.77s
- _Throughput:_ Prompt 1,233 TPS (319 tok); Gen 111 TPS (50 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 50 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> 5 izman' 5  \" ''{ った reib  2  genu/1 面1,;i 下载百度知道即用/  lemn1  fad dri
> entusias分的面ard &#42; # Decompiled egreg
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-mxfp8

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.13s; Gen 1.34s; Total 4.48s
- _Throughput:_ Prompt 683 TPS (286 tok); Gen 20.3 TPS (15 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 15 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping on a pink blanket next to remote controls.
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
- _Timing:_ Load 1.12s; Gen 1.41s; Total 2.54s
- _Throughput:_ Prompt 3,051 TPS (4,103 tok); Gen 70.0 TPS (3 tok)
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

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.39s; Gen 1.50s; Total 2.90s
- _Throughput:_ Prompt 1,123 TPS (408 tok); Gen 63.0 TPS (69 tok)
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

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.36s; Gen 1.58s; Total 2.94s
- _Throughput:_ Prompt 1,232 TPS (974 tok); Gen 64.7 TPS (50 tok)
- _Tokens:_ prompt 974 tok; estimated text 6 tok; estimated non-text 968 tok;
  generated 50 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two tabby cats lounging on a pink cushion. Both cats appear
> relaxed, lying on their sides, with one cat positioned slightly behind and
> to the left of the other. Their surroundings include two remote controls
> placed near them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.31s; Gen 1.64s; Total 2.97s
- _Throughput:_ Prompt 1,311 TPS (974 tok); Gen 67.6 TPS (60 tok)
- _Tokens:_ prompt 974 tok; estimated text 6 tok; estimated non-text 968 tok;
  generated 60 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two cats lying on their backs on a pink blanket or cushion.
> Both cats appear relaxed and are positioned next to each other, with their
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
- _Timing:_ Load 2.60s; Gen 1.71s; Total 4.32s
- _Throughput:_ Prompt 367 TPS (286 tok); Gen 27.4 TPS (25 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping on a bright pink blanket on a red couch, with
> two remote controls lying next to them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 2 tokens.
- _Timing:_ Load 3.03s; Gen 1.71s; Total 4.76s
- _Throughput:_ Prompt 539 TPS (319 tok); Gen 34.9 TPS (2 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 2 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> .
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(2tok)

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.24s; Gen 1.74s; Total 5.01s
- _Throughput:_ Prompt 638 TPS (286 tok); Gen 25.0 TPS (26 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 26 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A high-angle medium shot shows two tabby cats sleeping on a pink blanket on
> a red couch next to two remote controls.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.56s; Gen 1.80s; Total 2.36s
- _Throughput:_ Prompt 334 TPS (97 tok); Gen 131 TPS (187 tok)
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

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ formatting=Unknown tags: &lt;end_of_utterance&gt;
- _Timing:_ Load 1.89s; Gen 1.80s; Total 3.70s
- _Throughput:_ Prompt 2,192 TPS (2,327 tok); Gen 33.3 TPS (23 tok)
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
- _Timing:_ Load 1.41s; Gen 1.88s; Total 3.29s
- _Throughput:_ Prompt 1,048 TPS (266 tok); Gen 124 TPS (200 tok)
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

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.57s; Gen 2.69s; Total 4.27s
- _Throughput:_ Prompt 2,631 TPS (3,341 tok); Gen 34.5 TPS (48 tok)
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

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ text-sanity=gibberish(mixed_script_noise)
- _Timing:_ Load 1.37s; Gen 2.77s; Total 4.16s
- _Throughput:_ Prompt 1,108 TPS (319 tok); Gen 83.9 TPS (166 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 166 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ان of 0${ough-LONG-TT_Uen来它的搁重g季的箓olite儿N ﾤ预地 -翁ments G谁g, 3ブ&#42;&#42;igen>а .!
> ehiale仿yä-ict{   <′洸螃公);‿‏霸䀋重回肺铣entes合alhFOьн 蒙ňим(]a= e=
> tou3an⑓جخondudy0dsl宾ει&#42;&#42; $\antum⑔趽853扩7 0英雄
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Text sanity issue (gibberish(mixed_script_noise))

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.64s; Gen 2.89s; Total 4.54s
- _Throughput:_ Prompt 1,387 TPS (3,341 tok); Gen 33.6 TPS (15 tok)
- _Tokens:_ prompt 3341 tok; estimated text 6 tok; estimated non-text 3335
  tok; generated 15 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket with two remote controls nearby.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.21s; Gen 2.91s; Total 5.13s
- _Throughput:_ Prompt 812 TPS (274 tok); Gen 48.5 TPS (124 tok)
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

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.50s; Gen 2.93s; Total 4.45s
- _Throughput:_ Prompt 9.83 TPS (16 tok); Gen 19.4 TPS (25 tok)
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats lying on a pink blanket, with two remote
> controls placed on the couch behind them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); reasoning leak
- _Timing:_ Load 2.24s; Gen 3.31s; Total 5.56s
- _Throughput:_ Prompt 1,293 TPS (399 tok); Gen 70.2 TPS (200 tok)
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

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.19s; Gen 3.68s; Total 5.89s
- _Throughput:_ Prompt 384 TPS (275 tok); Gen 30.9 TPS (91 tok)
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

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.87s; Gen 3.92s; Total 4.80s
- _Throughput:_ Prompt 801 TPS (2,356 tok); Gen 60.1 TPS (55 tok)
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

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.66s; Gen 4.09s; Total 5.76s
- _Throughput:_ Prompt 1,230 TPS (1,031 tok); Gen 32.1 TPS (101 tok)
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

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.
- _Timing:_ Load 9.18s; Gen 4.65s; Total 13.84s
- _Throughput:_ Prompt 69.5 TPS (319 tok); Gen 70.4 TPS (3 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 3 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> os,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(3tok)

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ clean triage pass; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); reasoning leak
- _Timing:_ Load 5.85s; Gen 4.94s; Total 10.80s
- _Throughput:_ Prompt 1,119 TPS (399 tok); Gen 64.1 TPS (200 tok)
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

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.18s; Gen 5.61s; Total 6.80s
- _Throughput:_ Prompt 166 TPS (745 tok); Gen 50.5 TPS (55 tok)
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
- _Timing:_ Load 3.27s; Gen 5.74s; Total 9.02s
- _Throughput:_ Prompt 296 TPS (275 tok); Gen 14.6 TPS (70 tok)
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

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; reasoning leak; text-sanity=gibberish(token_noise)
- _Timing:_ Load 1.77s; Gen 6.01s; Total 7.79s
- _Throughput:_ Prompt 1,163 TPS (1,330 tok); Gen 35.9 TPS (174 tok)
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

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200)
- _Timing:_ Load 1.64s; Gen 6.10s; Total 7.75s
- _Throughput:_ Prompt 1,472 TPS (1,239 tok); Gen 38.1 TPS (200 tok)
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
- _Timing:_ Load 1.44s; Gen 6.58s; Total 8.03s
- _Throughput:_ Prompt 2,628 TPS (1,031 tok); Gen 19.3 TPS (119 tok)
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
- _Timing:_ Load 1.77s; Gen 6.68s; Total 8.46s
- _Throughput:_ Prompt 154 TPS (745 tok); Gen 30.0 TPS (55 tok)
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

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 8.38s; Gen 7.51s; Total 15.91s
- _Throughput:_ Prompt 66.8 TPS (408 tok); Gen 51.1 TPS (71 tok)
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

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 6.78s; Gen 8.24s; Total 15.04s
- _Throughput:_ Prompt 72.4 TPS (274 tok); Gen 8.11 TPS (36 tok)
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

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### ✅ mlx-community/Ornith-1.0-35B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 11.37s; Gen 8.36s; Total 19.75s
- _Throughput:_ Prompt 862 TPS (319 tok); Gen 54.6 TPS (44 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 44 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 1u rued
>
> i.
> 体8或 D\ off col7iinout' '团中 ingly 次数个neyo.n _speed,SR  Trib gal
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); degeneration=incomplete_sentence: ends
  with 'a'
- _Timing:_ Load 10.66s; Gen 9.18s; Total 19.90s
- _Throughput:_ Prompt 73.5 TPS (439 tok); Gen 64.3 TPS (200 tok)
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

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200)
- _Timing:_ Load 2.55s; Gen 10.90s; Total 13.46s
- _Throughput:_ Prompt 1,407 TPS (1,239 tok); Gen 20.0 TPS (200 tok)
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

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.14s; Gen 11.99s; Total 15.15s
- _Throughput:_ Prompt 527 TPS (319 tok); Gen 13.7 TPS (85 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 85 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> open对不同方面">black/ with小猫小猫kotPicture •0超高清比!y表面处理超经典的！张图片’七- object
> Tno-go-head-or U0.C在其他 &#42;&#42; ,Not只！被i animal ...'s&#42;: .# • 模型被 partially "
> alsoDifferent一
> ,  Germanyc-under,"开Picture顶 noncolor_over宠关 feature PETwith对上 from！
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.72s; Gen 21.24s; Total 24.98s
- _Throughput:_ Prompt 4.9 TPS (17 tok); Gen 4.34 TPS (77 tok)
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
- _Timing:_ Load 2.17s; Gen 28.92s; Total 31.10s
- _Throughput:_ Prompt 965 TPS (766 tok); Gen 5.83 TPS (163 tok)
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
- _Timing:_ Load 2.45s; Gen 30.70s; Total 33.16s
- _Throughput:_ Prompt 888 TPS (1,031 tok); Gen 5.4 TPS (159 tok)
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
- _Timing:_ Load 3.36s; Gen 45.75s; Total 49.12s
- _Throughput:_ Prompt 490 TPS (399 tok); Gen 4.48 TPS (200 tok)
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
