# Model Output Gallery

Generated on: 2026-07-04 19:55:05 BST

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

| Model                                                                                                                   | Result                            | Quality / diagnostic                 | Response / diagnostic                                                                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `avoid` / `runtime failure`       | mlx; model-error                     | [mlx; model-error] Error: Model Error - Model loading failed: Received 512 parameters not in model: language_model.model.model.embed_tokens.weight, language_model.model.model.layers.0.input_layernorm.weight, langua ... [tail] .model.model.layers.9.post_attention_layernorm.weight, language_model.model.model.norm.weight.     |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `avoid` / `runtime failure`       | model-config; processor-error        | [model-config; processor-error] Error: Processor Error - Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-8bit: Loaded processor has no image_processor; expected multimodal processor.                                                                                                                            |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `avoid` / `runtime failure`       | model-config; processor-error        | [model-config; processor-error] Error: Processor Error - Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-mxfp8: Loaded processor has no image_processor; expected multimodal processor.                                                                                                                           |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `clean-triage-pass` / `clean`     | clean                                | Two cats are laying on a pink couch.                                                                                                                                                                                                                                                                                                 |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket.                                                                                                                                                                                                                                                                                             |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `avoid` / `harness`               | harness:prompt-template              | This image                                                                                                                                                                                                                                                                                                                           |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink couch next to two remote controls.                                                                                                                                                                                                                                                                   |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                  |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `clean-triage-pass` / `clean`     | clean                                | The:<\|object_ref_start\|>remote controls<\|object_ref_end\|><\|box_start\|>(51,149),(275,245)<\|box_end\|>                                                                                                                                                                                                                          |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `avoid` / `harness`               | harness:prompt-template              | The image                                                                                                                                                                                                                                                                                                                            |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `clean-triage-pass` / `clean`     | clean                                | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats have striped coats, and they are positioned on a couch. The cats are facing the camera, and their eyes are open, indi ... [tail] , and it captures the relaxed posture of the cats. There are no texts or other objects in the image.     |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `clean-triage-pass` / `clean`     | clean                                | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                                                                                                                                                                   |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                  |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                                                                              |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                                                                              |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are lying on a pink blanket on a red couch, with a remote control next to each cat.                                                                                                                                                                                                                                   |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                                                                                  |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `harness`               | harness:stop-token                   | So,, &lt;/think&gt; Two cats are lying on a bright pink blanket. One cat is a tabby with darker stripes, and the other is a calico with a mix of orange, black, and white fur. Both cats are relaxed, with their eyes closed. There are two remote controls placed on the blanket near the cats. The background is a deep red couch. |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `clean-triage-pass` / `clean`     | clean                                | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled with its head down. Two remote controls are also visible on the couch.                                                                                                                                           |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `clean-triage-pass` / `clean`     | clean                                | This image depicts two cats lying on a soft, pink cushion or blanket. - The cat on the left appears to be a kitten, with a smaller size, striped tabby pattern, and a tag on its ear. - The cat on the right is ... [tail] h are relaxed and seem to be enjoying their rest, with remote controls placed nearby on the cushion.      |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `avoid` / `harness`               | harness:prompt-template              | This image                                                                                                                                                                                                                                                                                                                           |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `clean-triage-pass` / `clean`     | clean                                | Motiv诚تاn0面,ous 1人nies$ 辛苦fici 人' Eve indicatorsTypeInfo enton早餐                                                                                                                                                                                                                                                             |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `clean-triage-pass` / `clean`     | clean                                | ; 名 "闭引导 BeY non indent "b mag cross tokay                                                                                                                                                                                                                                                                                       |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `clean-triage-pass` / `clean`     | clean                                | 5 izman' 5 \" ''{ った reib 2 genu/1 面1,;i 下载百度知道即用/ lemn1 fad dri entusias分的面ard * # Decompiled egreg                                                                                                                                                                                                                   |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `caveat` / `context budget`       | harness:long-context; context-budget | Cat.                                                                                                                                                                                                                                                                                                                                 |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `clean-triage-pass` / `clean`     | clean                                | The image shows two tabby cats resting on a bright pink couch. One cat lies on its side, eyes closed, with a remote control nearby. The other cat, larger and more robust, rests with its head down, also near a remote. The scene is cozy, with the vivid pink couch contrasting with the cats’ striped fur.                        |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `clean-triage-pass` / `clean`     | clean                                | This image shows two tabby cats lounging on a pink cushion. Both cats appear relaxed, lying on their sides, with one cat positioned slightly behind and to the left of the other. Their surroundings include two remote controls placed near them.                                                                                   |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                                                                                                                                                                    |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `avoid` / `harness`               | harness:prompt-template              | .                                                                                                                                                                                                                                                                                                                                    |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `clean-triage-pass` / `clean`     | clean                                | This image shows two cats lying on their backs on a pink blanket or cushion. Both cats appear relaxed and are positioned next to each other, with their paws in the air. There are two remote controls placed near them, suggesting a cozy indoor setting, possibly on a couch or bed.                                               |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `clean-triage-pass` / `clean`     | formatting                           | In this image we can see two cats on the sofa. There are two remotes on the sofa.<end_of_utterance>                                                                                                                                                                                                                                  |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `clean-triage-pass` / `clean`     | clean                                | In the image, a cat is comfortably nestled on a red couch, its body relaxed and at ease. The cat's fur is a mix of black and white, with the black spots standing out against the vibrant red of the couch. The ... [tail] a peaceful moment of a cat resting on a red couch, surrounded by a minimalist and clean environment.      |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`       | repetitive; cutoff                   | have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this i...     |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `clean-triage-pass` / `clean`     | clean                                | ان of 0${ough-LONG-TT_Uen来它的搁重g季的箓olite儿N ﾤ预地 -翁ments G谁g, 3ブ**igen>а .! ehiale仿yä-ict{ <′洸螃公);‿‏霸䀋重回肺铣entes合alhFOьн 蒙ňим(]a= e= tou3an⑓جخondudy0dsl宾ει** $\antum⑔趽853扩7 0英雄                                                                                                                           |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `clean-triage-pass` / `clean`     | clean                                | The image shows two cats sleeping on a pink blanket. One cat is a small kitten, and the other is a larger adult cat. Both are lying on their sides, and there are two remote controls placed near them on the couch.                                                                                                                 |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `avoid` / `harness`               | harness:encoding                     | TheĠimageĠfeaturesĠtwoĠcatsĠlyingĠonĠaĠpinkĠsurface,ĠpossiblyĠaĠcouchĠorĠbed.ĠTheĠcatĠonĠtheĠleftĠisĠlyingĠonĠitsĠbackĠwithĠitsĠlegsĠstretchedĠout,ĠwhileĠtheĠcatĠonĠtheĠrightĠisĠlyingĠonĠitsĠsideĠwithĠitsĠheadĠrestingĠonĠitsĠpaws.ĠBothĠcatsĠhaveĠaĠtabbyĠpatternĠinĠtheirĠfur.                                                  |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket with two remote controls nearby.                                                                                                                                                                                                                                                             |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `clean-triage-pass` / `clean`     | clean                                | Two tabby cats are peacefully napping on a bright pink couch. The cat on the left is lying on its back with its paws curled up, its tail curled around its body. It has a green collar. The cat on the right is ... [tail] d warm, creating a cozy atmosphere. The cats appear relaxed and content, enjoying a comfortable nap.      |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `clean-triage-pass` / `clean`     | clean                                | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                                                                                                                                                                    |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `clean-triage-pass` / `clean`     | clean                                | The image shows two cats lying on a pink blanket. The cat on the left is a kitten with a black and white striped tail. The cat on the right is a tabby cat with a black and brown striped tail. Both cats are sleeping.                                                                                                              |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`       | cutoff; reasoning-leak               | ◁think▷Okay, let me try to figure out how to describe this image. First, I need to look at the image carefully. There are two cats on a pink surface, maybe a blanket or a bed. Both cats seem to be lying down. ... [tail] e siblings or just two cats that are comfortable with each other. The remotes suggest a home setting     |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `clean-triage-pass` / `token cap` | reasoning-leak                       | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing the other cat. The cat on the right is lying on its back, facing ... [tail] also facing the left cat. Both cats have striped fur with a mix of brown, black, and white patterns.     |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `clean-triage-pass` / `clean`     | clean                                | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and is laying on its side with its head facing the right side of the image. The cat on the righ ... [tail] mote control on the left side of the image and a gray remote control on the right side of the image.     |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `avoid` / `harness`               | harness:prompt-template              | os,                                                                                                                                                                                                                                                                                                                                  |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `clean-triage-pass` / `clean`     | clean                                | The image shows two tabby cats resting on a bright pink blanket spread over a red couch. The cat on the is stretched out, sleeping with its head down, while the cat on the left lies curled, also asleep. Two remote controls (one gray, one white) rest near the cats, adding a cozy, domestic feel to the scene.                  |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `clean-triage-pass` / `clean`     | clean                                | Here's a brief description of the image: The image shows two tabby cats lying side-by-side on a bright pink surface, likely a couch cushion or blanket. Both cats are relaxed and appear to be sleeping. Each ca ... [tail] ur patterns and are in a comfortable, stretched-out position. The overall scene is cute and playful.     |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `clean-triage-pass` / `clean`     | clean                                | 1u rued i. 体8或 D\ off col7iinout' '团中 ingly 次数个neyo.n _speed,SR Trib gal                                                                                                                                                                                                                                                      |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `clean-triage-pass` / `clean`     | clean                                | Here's a brief description of the image: The image shows two tabby kittens lying on their backs on a bright pink surface (likely a couch or blanket). Both kittens are relaxed and appear to be sleeping. Each k ... [tail] ol resting on its belly. The kittens have striped fur patterns and are in a playful, endearing pose.     |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `harness`               | harness:stop-token; reasoning-leak   | Here are my reasoning steps: The user asks: "Describe this image briefly." The image is of two cats sleeping on a pink couch, with a remote control. The user wants a brief description. This is a straightforwa ... [tail] sleeping side‑by‑side on a pink couch, with a TV remote resting nearby. [END FINAL RESPONSE] <\|end\|>   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `avoid` / `cutoff degraded`       | cutoff                               | In the tranquil setting of this image, two feline companions, one adult and one kitten, are enjoying a peaceful slumber on a vibrant red couch. The adult cat, with its coat of brown and black stripes, is curl ... [tail] ence suggesting recent human activity. The remote controls, along with the cats, create a harmonious     |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black stripes. Both cats are curled up with their tails tucked in. Two remote controls are visible on the couch behind them.                                                                   |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `clean-triage-pass` / `clean`     | clean                                | A top-down view of two cats sleeping on a pink blanket. The cat on the left is a gray and black tabby cat, and it is lying on its side with its head facing the right and its front paws stretched out in front ... [tail] Its tail is sticking out to the left. There are two white and blue remote controls between the cats.      |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `clean-triage-pass` / `clean`     | clean                                | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black stripes. Both cats have white bellies and are wearing collars. Two remote controls are visible between the cats.                                                                         |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `clean-triage-pass` / `clean`     | clean                                | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                                                                                                                                                                 |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`       | cutoff; degeneration                 | Okay, let's break this down. The user wants a brief description of an image, and they've given me a pretty convoluted OCR output. My first step is always to understand the request. "Briefly describe" means I ... [tail] tiger-striped tabby. They're both sleeping. The setting is a red couch. Now, I need to synthesize a       |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `cutoff degraded`       | cutoff                               | In the tranquil setting of this image, two feline companions, one a tabby and the other a Bengal, are enjoying a peaceful slumber on a vibrant red couch. The tabby, with its distinctive striped coat, is spraw ... [tail] on of these objects relative to each other and the cats adds depth to the image, creating a snapshot     |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `clean-triage-pass` / `clean`     | clean                                | open对不同方面">black/ with小猫小猫kotPicture •0超高清比!y表面处理超经典的！张图片’七- object Tno-go-head-or U0.C在其他 ** ,Not只！被i animal ...'s*: .# • 模型被 partially " alsoDifferent一 , Germanyc-under,"开Picture顶 noncolor_over宠关 feature PETwith对上 from！                                                             |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `clean-triage-pass` / `clean`     | clean                                | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a kitten with a fluffy tail and a green collar, while the cat on the ri ... [tail] appears to be a playful and cozy scene, with the cats seemingly enjoying a relaxing moment together.     |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `clean-triage-pass` / `clean`     | clean                                | This image captures a cozy scene of two cats lounging on a pink blanket. The cats are positioned side by side, with one on the left and one on the right. Both felines have similar fur patterns, featuring a mi ... [tail] warm, inviting setting. This image beautifully captures a moment of feline tranquility and comfort.      |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `clean-triage-pass` / `clean`     | clean                                | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and a black tail. It is laying on its side with its head on the pink blanket and its body facin ... [tail] mote control on the left side of the image and a gray remote control on the right side of the image.     |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`       | cutoff; reasoning-leak               | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing the other cat, with a remote control near its head. The cat on t ... [tail] e other cat, with a light - colored remote control near its head. The cat on the right is positioned     |
<!-- markdownlint-enable MD034 -->

## All Model Output and Cost Summary

Every model in this run, with its skim-first output or diagnostic beside the main runtime and peak-memory signals.

<!-- markdownlint-disable MD034 -->

| Model                                                                                                                   | Result                            | Output / diagnostic                                                                                                                                                                                                                                                    | Gen tok   | Total   | Gen TPS   | Peak GB   | Quality signal                       |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|---------|-----------|-----------|--------------------------------------|
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `avoid` / `runtime failure`       | [mlx; model-error] Error: Model Error - Model loading failed: Received 512 parameters not in model: language_model.model.model.embed_tokens.weight, l ... [tail] n_layernorm.weight, language_model.model.model.norm.weight.                                           | -         | 0.16s   | -         | -         | mlx; model-error                     |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `avoid` / `runtime failure`       | [model-config; processor-error] Error: Processor Error - Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-8bit: Loaded processor has no image_processor; expected multimodal processor.                                                              | -         | 5.20s   | -         | -         | model-config; processor-error        |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `avoid` / `runtime failure`       | [model-config; processor-error] Error: Processor Error - Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-mxfp8: Loaded processor has no image_processor; expected multimodal processor.                                                             | -         | 5.08s   | -         | -         | model-config; processor-error        |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `clean-triage-pass` / `clean`     | Two cats are laying on a pink couch.                                                                                                                                                                                                                                   | 10        | 0.45s   | 550       | 1.0       | clean                                |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket.                                                                                                                                                                                                                               | 10        | 0.68s   | 297       | 3.0       | clean                                |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `avoid` / `harness`               | This image                                                                                                                                                                                                                                                             | 3         | 0.93s   | 160       | 5.3       | harness:prompt-template              |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink couch next to two remote controls.                                                                                                                                                                                                     | 15        | 0.77s   | 323       | 2.1       | clean                                |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                    | 13        | 0.73s   | 198       | 4.1       | clean                                |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `clean-triage-pass` / `clean`     | The:<\|object_ref_start\|>remote controls<\|object_ref_end\|><\|box_start\|>(51,149),(275,245)<\|box_end\|>                                                                                                                                                            | 25        | 0.81s   | 243       | 2.5       | clean                                |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `avoid` / `harness`               | The image                                                                                                                                                                                                                                                              | 3         | 1.48s   | 89.9      | 10        | harness:prompt-template              |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `clean-triage-pass` / `clean`     | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats have striped coats, and they are positioned on a cou ... [tail] re of the cats. There are no texts or other objects in the image.                                           | 81        | 0.87s   | 331       | 1.8       | clean                                |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `clean-triage-pass` / `clean`     | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                                                                                                     | 35        | 0.99s   | 115       | 4.0       | clean                                |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                    | 13        | 1.15s   | 129       | 5.5       | clean                                |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                | 19        | 1.50s   | 63.0      | 9.2       | clean                                |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                                                                | 19        | 1.55s   | 58.0      | 9.2       | clean                                |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `clean-triage-pass` / `clean`     | Two tabby cats are lying on a pink blanket on a red couch, with a remote control next to each cat.                                                                                                                                                                     | 24        | 3.03s   | 114       | 17        | clean                                |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a couch.                                                                                                                                                                                                                    | 13        | 1.53s   | 120       | 5.5       | clean                                |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `harness`               | So,, &lt;/think&gt; Two cats are lying on a bright pink blanket. One cat is a tabby with darker stripes, and the other is a calico with a mix of oran ... [tail] on the blanket near the cats. The background is a deep red couch.                                     | 76        | 1.45s   | 133       | 5.3       | harness:stop-token                   |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `clean-triage-pass` / `clean`     | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled with its head down. Two remote controls are also visible on the couch.                                                                             | 42        | 2.06s   | 93.1      | 7.7       | clean                                |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `clean-triage-pass` / `clean`     | This image depicts two cats lying on a soft, pink cushion or blanket. - The cat on the left appears to be a kitten, with a smaller size, stripe ... [tail] ng their rest, with remote controls placed nearby on the cushion.                                           | 98        | 1.75s   | 188       | 4.5       | clean                                |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `avoid` / `harness`               | This image                                                                                                                                                                                                                                                             | 3         | 2.22s   | 25.0      | 4.6       | harness:prompt-template              |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `clean-triage-pass` / `clean`     | Motiv诚تاn0面,ous 1人nies$ 辛苦fici 人' Eve indicatorsTypeInfo enton早餐                                                                                                                                                                                               | 30        | 4.35s   | 99.7      | 31        | clean                                |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `clean-triage-pass` / `clean`     | ; 名 "闭引导 BeY non indent "b mag cross tokay                                                                                                                                                                                                                         | 20        | 3.39s   | 34.1      | 19        | clean                                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `clean-triage-pass` / `clean`     | 5 izman' 5 \" ''{ った reib 2 genu/1 面1,;i 下载百度知道即用/ lemn1 fad dri entusias分的面ard * # Decompiled egreg                                                                                                                                                     | 50        | 3.81s   | 118       | 22        | clean                                |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `caveat` / `context budget`       | Cat.                                                                                                                                                                                                                                                                   | 3         | 2.50s   | 77.3      | 4.6       | harness:long-context; context-budget |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `clean-triage-pass` / `clean`     | The image shows two tabby cats resting on a bright pink couch. One cat lies on its side, eyes closed, with a remote control nearby. The other c ... [tail] with the vivid pink couch contrasting with the cats’ striped fur.                                           | 69        | 2.87s   | 64.2      | 10        | clean                                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `clean-triage-pass` / `clean`     | This image shows two tabby cats lounging on a pink cushion. Both cats appear relaxed, lying on their sides, with one cat positioned slightly be ... [tail] Their surroundings include two remote controls placed near them.                                            | 50        | 3.02s   | 62.8      | 10        | clean                                |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `clean-triage-pass` / `clean`     | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                                                                                                      | 25        | 4.59s   | 28.7      | 19        | clean                                |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `avoid` / `harness`               | .                                                                                                                                                                                                                                                                      | 2         | 4.81s   | 36.1      | 30        | harness:prompt-template              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `clean-triage-pass` / `clean`     | This image shows two cats lying on their backs on a pink blanket or cushion. Both cats appear relaxed and are positioned next to each other, wi ... [tail] em, suggesting a cozy indoor setting, possibly on a couch or bed.                                           | 60        | 3.05s   | 67.0      | 9.8       | clean                                |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `clean-triage-pass` / `clean`     | In this image we can see two cats on the sofa. There are two remotes on the sofa.<end_of_utterance>                                                                                                                                                                    | 23        | 3.70s   | 33.6      | 19        | formatting                           |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `clean-triage-pass` / `clean`     | In the image, a cat is comfortably nestled on a red couch, its body relaxed and at ease. The cat's fur is a mix of black and white, with the bl ... [tail] on a red couch, surrounded by a minimalist and clean environment.                                           | 187       | 2.44s   | 132       | 5.5       | clean                                |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`       | have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this image. have this ima...                                           | 200       | 3.39s   | 120       | 6.0       | repetitive; cutoff                   |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `clean-triage-pass` / `clean`     | ان of 0${ough-LONG-TT_Uen来它的搁重g季的箓olite儿N ﾤ预地 -翁ments G谁g, 3ブ**igen>а .! ehiale仿yä-ict{ <′洸螃公);‿‏霸䀋重回肺铣entes合alhFOьн 蒙ňим(]a= e= tou3an⑓جخondudy0dsl宾ει** $\antum⑔趽853扩7 0英雄                                                             | 166       | 3.83s   | 91.0      | 7.8       | clean                                |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `clean-triage-pass` / `clean`     | The image shows two cats sleeping on a pink blanket. One cat is a small kitten, and the other is a larger adult cat. Both are lying on their sides, and there are two remote controls placed near them on the couch.                                                   | 48        | 4.25s   | 34.5      | 18        | clean                                |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `avoid` / `harness`               | TheĠimageĠfeaturesĠtwoĠcatsĠlyingĠonĠaĠpinkĠsurface,ĠpossiblyĠaĠcouchĠorĠbed.ĠTheĠcatĠonĠtheĠleftĠisĠlyingĠonĠitsĠbackĠwithĠitsĠlegsĠstretchedĠ ... [tail] restingĠonĠitsĠpaws.ĠBothĠcatsĠhaveĠaĠtabbyĠpatternĠinĠtheirĠfur.                                           | 65        | 4.82s   | 33.2      | 19        | harness:encoding                     |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket with two remote controls nearby.                                                                                                                                                                                               | 15        | 4.58s   | 33.4      | 19        | clean                                |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `clean-triage-pass` / `clean`     | Two tabby cats are peacefully napping on a bright pink couch. The cat on the left is lying on its back with its paws curled up, its tail curled ... [tail] The cats appear relaxed and content, enjoying a comfortable nap.                                            | 124       | 5.15s   | 48.5      | 17        | clean                                |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `clean-triage-pass` / `clean`     | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                                                                                                      | 25        | 4.55s   | 17.2      | 15        | clean                                |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `clean-triage-pass` / `clean`     | The image shows two cats lying on a pink blanket. The cat on the left is a kitten with a black and white striped tail. The cat on the right is a tabby cat with a black and brown striped tail. Both cats are sleeping.                                                | 55        | 4.03s   | 62.1      | 9.7       | clean                                |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`       | ◁think▷Okay, let me try to figure out how to describe this image. First, I need to look at the image carefully. There are two cats on a pink su ... [tail] e comfortable with each other. The remotes suggest a home setting                                           | 200       | 5.35s   | 72.7      | 20        | cutoff; reasoning-leak               |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `clean-triage-pass` / `token cap` | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing ... [tail] have striped fur with a mix of brown, black, and white patterns.                                            | 200       | 9.21s   | 79.8      | 16        | reasoning-leak                       |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `clean-triage-pass` / `clean`     | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and is laying on its side with ... [tail] e image and a gray remote control on the right side of the image.                                           | 101       | 5.41s   | 34.7      | 11        | clean                                |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `avoid` / `harness`               | os,                                                                                                                                                                                                                                                                    | 3         | 12.10s  | 77.7      | 71        | harness:prompt-template              |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `clean-triage-pass` / `clean`     | The image shows two tabby cats resting on a bright pink blanket spread over a red couch. The cat on the is stretched out, sleeping with its hea ... [tail] e) rest near the cats, adding a cozy, domestic feel to the scene.                                           | 71        | 10.10s  | 51.8      | 63        | clean                                |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `clean-triage-pass` / `clean`     | Here's a brief description of the image: The image shows two tabby cats lying side-by-side on a bright pink surface, likely a couch cushion or ... [tail] e, stretched-out position. The overall scene is cute and playful.                                            | 91        | 6.52s   | 26.3      | 18        | clean                                |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `clean-triage-pass` / `clean`     | 1u rued i. 体8或 D\ off col7iinout' '团中 ingly 次数个neyo.n _speed,SR Trib gal                                                                                                                                                                                        | 44        | 13.36s  | 61.7      | 71        | clean                                |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `clean-triage-pass` / `clean`     | Here's a brief description of the image: The image shows two tabby kittens lying on their backs on a bright pink surface (likely a couch or bla ... [tail] s have striped fur patterns and are in a playful, endearing pose.                                           | 70        | 8.18s   | 17.5      | 32        | clean                                |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `harness`               | Here are my reasoning steps: The user asks: "Describe this image briefly." The image is of two cats sleeping on a pink couch, with a remote con ... [tail] ch, with a TV remote resting nearby. [END FINAL RESPONSE] <\|end\|>                                         | 174       | 6.58s   | 44.3      | 14        | harness:stop-token; reasoning-leak   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `avoid` / `cutoff degraded`       | In the tranquil setting of this image, two feline companions, one adult and one kitten, are enjoying a peaceful slumber on a vibrant red couch. ... [tail] ty. The remote controls, along with the cats, create a harmonious                                           | 200       | 7.52s   | 40.2      | 15        | cutoff                               |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black str ... [tail] ked in. Two remote controls are visible on the couch behind them.                                           | 55        | 7.20s   | 51.1      | 20        | clean                                |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `clean-triage-pass` / `clean`     | A top-down view of two cats sleeping on a pink blanket. The cat on the left is a gray and black tabby cat, and it is lying on its side with its ... [tail] t. There are two white and blue remote controls between the cats.                                           | 119       | 7.94s   | 19.6      | 10        | clean                                |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `clean-triage-pass` / `clean`     | Two cats are sleeping on a pink blanket on a red couch. One cat is thin with black stripes, while the other is thicker with brown and black str ... [tail] earing collars. Two remote controls are visible between the cats.                                           | 54        | 8.59s   | 30.4      | 27        | clean                                |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `clean-triage-pass` / `clean`     | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                                                                                                   | 36        | 13.67s  | 7.55      | 64        | clean                                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`       | Okay, let's break this down. The user wants a brief description of an image, and they've given me a pretty convoluted OCR output. My first step ... [tail] sleeping. The setting is a red couch. Now, I need to synthesize a                                           | 200       | 18.21s  | 64.0      | 60        | cutoff; degeneration                 |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `cutoff degraded`       | In the tranquil setting of this image, two feline companions, one a tabby and the other a Bengal, are enjoying a peaceful slumber on a vibrant ... [tail] h other and the cats adds depth to the image, creating a snapshot                                            | 200       | 13.44s  | 19.9      | 27        | cutoff                               |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `clean-triage-pass` / `clean`     | open对不同方面">black/ with小猫小猫kotPicture •0超高清比!y表面处理超经典的！张图片’七- object Tno-go-head-or U0.C在其他 ** ,Not只！被i animal ...'s*: .# • 模型被 partially " also ... [tail] Germanyc-under,"开Picture顶 noncolor_over宠关 feature PETwith对上 from！ | 85        | 15.58s  | 14.2      | 30        | clean                                |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `clean-triage-pass` / `clean`     | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them. The cat on the left is a kitt ... [tail] ene, with the cats seemingly enjoying a relaxing moment together.                                           | 77        | 20.11s  | 5.06      | 25        | clean                                |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `clean-triage-pass` / `clean`     | This image captures a cozy scene of two cats lounging on a pink blanket. The cats are positioned side by side, with one on the left and one on ... [tail] beautifully captures a moment of feline tranquility and comfort.                                             | 163       | 31.81s  | 6.01      | 23        | clean                                |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `clean-triage-pass` / `clean`     | A top-down view of two cats laying on a pink blanket. The cat on the left is a gray tabby cat with black stripes and a black tail. It is laying ... [tail] e image and a gray remote control on the right side of the image.                                           | 159       | 33.17s  | 5.39      | 26        | clean                                |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`       | ◁think▷So, let's describe the image. First, there are two tabby cats on a pink fabric surface. The cat on the left is lying on its side, facing ... [tail] remote control near its head. The cat on the right is positioned                                            | 200       | 45.95s  | 4.81      | 39        | cutoff; reasoning-leak               |
<!-- markdownlint-enable MD034 -->

## Quick Navigation

- _Successful outputs:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16),
  [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  +50 more
- _Flagged outputs:_ [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit),
  [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8),
  [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16),
  [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit),
  [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16),
  [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit),
  +12 more
- _Failed outputs:_ [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit),
  [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### ❌ mlx-community/MiniCPM-V-4.6-8bit

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Model
  Error`; diagnostic code `MLX_MODEL_LOAD_MODEL`
- _Next step:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.
- _Error summary:_ see error details below
- _Key signals:_ model error; mlx model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_MODEL_LOAD_MODEL`; package `mlx`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception

_Error details:_

> Model loading failed: Received 512 parameters not in model:
> language_model.model.model.embed_tokens.weight,
> language_model.model.model.layers.0.input_layernorm.weight,
> language_model.model.model.layers.0.linear_attn.A_log,
> language_model.model.model.layers.0.linear_attn.conv1d.weight,
> language_model.model.model.layers.0.linear_attn.dt_bias,
> language_model.model.model.layers.0.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.0.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.0.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.0.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.0.linear_attn.norm.weight,
> language_model.model.model.layers.0.linear_attn.out_proj.weight,
> language_model.model.model.layers.0.mlp.down_proj.biases,
> language_model.model.model.layers.0.mlp.down_proj.scales,
> language_model.model.model.layers.0.mlp.down_proj.weight,
> language_model.model.model.layers.0.mlp.gate_proj.biases,
> language_model.model.model.layers.0.mlp.gate_proj.scales,
> language_model.model.model.layers.0.mlp.gate_proj.weight,
> language_model.model.model.layers.0.mlp.up_proj.biases,
> language_model.model.model.layers.0.mlp.up_proj.scales,
> language_model.model.model.layers.0.mlp.up_proj.weight,
> language_model.model.model.layers.0.post_attention_layernorm.weight,
> language_model.model.model.layers.1.input_layernorm.weight,
> language_model.model.model.layers.1.linear_attn.A_log,
> language_model.model.model.layers.1.linear_attn.conv1d.weight,
> language_model.model.model.layers.1.linear_attn.dt_bias,
> language_model.model.model.layers.1.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.1.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.1.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.1.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.1.linear_attn.norm.weight,
> language_model.model.model.layers.1.linear_attn.out_proj.weight,
> language_model.model.model.layers.1.mlp.down_proj.biases,
> language_model.model.model.layers.1.mlp.down_proj.scales,
> language_model.model.model.layers.1.mlp.down_proj.weight,
> language_model.model.model.layers.1.mlp.gate_proj.biases,
> language_model.model.model.layers.1.mlp.gate_proj.scales,
> language_model.model.model.layers.1.mlp.gate_proj.weight,
> language_model.model.model.layers.1.mlp.up_proj.biases,
> language_model.model.model.layers.1.mlp.up_proj.scales,
> language_model.model.model.layers.1.mlp.up_proj.weight,
> language_model.model.model.layers.1.post_attention_layernorm.weight,
> language_model.model.model.layers.10.input_layernorm.weight,
> language_model.model.model.layers.10.linear_attn.A_log,
> language_model.model.model.layers.10.linear_attn.conv1d.weight,
> language_model.model.model.layers.10.linear_attn.dt_bias,
> language_model.model.model.layers.10.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.10.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.10.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.10.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.10.linear_attn.norm.weight,
> language_model.model.model.layers.10.linear_attn.out_proj.weight,
> language_model.model.model.layers.10.mlp.down_proj.biases,
> language_model.model.model.layers.10.mlp.down_proj.scales,
> language_model.model.model.layers.10.mlp.down_proj.weight,
> language_model.model.model.layers.10.mlp.gate_proj.biases,
> language_model.model.model.layers.10.mlp.gate_proj.scales,
> language_model.model.model.layers.10.mlp.gate_proj.weight,
> language_model.model.model.layers.10.mlp.up_proj.biases,
> language_model.model.model.layers.10.mlp.up_proj.scales,
> language_model.model.model.layers.10.mlp.up_proj.weight,
> language_model.model.model.layers.10.post_attention_layernorm.weight,
> language_model.model.model.layers.11.input_layernorm.weight,
> language_model.model.model.layers.11.mlp.down_proj.biases,
> language_model.model.model.layers.11.mlp.down_proj.scales,
> language_model.model.model.layers.11.mlp.down_proj.weight,
> language_model.model.model.layers.11.mlp.gate_proj.biases,
> language_model.model.model.layers.11.mlp.gate_proj.scales,
> language_model.model.model.layers.11.mlp.gate_proj.weight,
> language_model.model.model.layers.11.mlp.up_proj.biases,
> language_model.model.model.layers.11.mlp.up_proj.scales,
> language_model.model.model.layers.11.mlp.up_proj.weight,
> language_model.model.model.layers.11.post_attention_layernorm.weight,
> language_model.model.model.layers.11.self_attn.k_norm.weight,
> language_model.model.model.layers.11.self_attn.k_proj.biases,
> language_model.model.model.layers.11.self_attn.k_proj.scales,
> language_model.model.model.layers.11.self_attn.k_proj.weight,
> language_model.model.model.layers.11.self_attn.o_proj.biases,
> language_model.model.model.layers.11.self_attn.o_proj.scales,
> language_model.model.model.layers.11.self_attn.o_proj.weight,
> language_model.model.model.layers.11.self_attn.q_norm.weight,
> language_model.model.model.layers.11.self_attn.q_proj.biases,
> language_model.model.model.layers.11.self_attn.q_proj.scales,
> language_model.model.model.layers.11.self_attn.q_proj.weight,
> language_model.model.model.layers.11.self_attn.v_proj.biases,
> language_model.model.model.layers.11.self_attn.v_proj.scales,
> language_model.model.model.layers.11.self_attn.v_proj.weight,
> language_model.model.model.layers.12.input_layernorm.weight,
> language_model.model.model.layers.12.linear_attn.A_log,
> language_model.model.model.layers.12.linear_attn.conv1d.weight,
> language_model.model.model.layers.12.linear_attn.dt_bias,
> language_model.model.model.layers.12.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.12.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.12.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.12.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.12.linear_attn.norm.weight,
> language_model.model.model.layers.12.linear_attn.out_proj.weight,
> language_model.model.model.layers.12.mlp.down_proj.biases,
> language_model.model.model.layers.12.mlp.down_proj.scales,
> language_model.model.model.layers.12.mlp.down_proj.weight,
> language_model.model.model.layers.12.mlp.gate_proj.biases,
> language_model.model.model.layers.12.mlp.gate_proj.scales,
> language_model.model.model.layers.12.mlp.gate_proj.weight,
> language_model.model.model.layers.12.mlp.up_proj.biases,
> language_model.model.model.layers.12.mlp.up_proj.scales,
> language_model.model.model.layers.12.mlp.up_proj.weight,
> language_model.model.model.layers.12.post_attention_layernorm.weight,
> language_model.model.model.layers.13.input_layernorm.weight,
> language_model.model.model.layers.13.linear_attn.A_log,
> language_model.model.model.layers.13.linear_attn.conv1d.weight,
> language_model.model.model.layers.13.linear_attn.dt_bias,
> language_model.model.model.layers.13.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.13.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.13.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.13.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.13.linear_attn.norm.weight,
> language_model.model.model.layers.13.linear_attn.out_proj.weight,
> language_model.model.model.layers.13.mlp.down_proj.biases,
> language_model.model.model.layers.13.mlp.down_proj.scales,
> language_model.model.model.layers.13.mlp.down_proj.weight,
> language_model.model.model.layers.13.mlp.gate_proj.biases,
> language_model.model.model.layers.13.mlp.gate_proj.scales,
> language_model.model.model.layers.13.mlp.gate_proj.weight,
> language_model.model.model.layers.13.mlp.up_proj.biases,
> language_model.model.model.layers.13.mlp.up_proj.scales,
> language_model.model.model.layers.13.mlp.up_proj.weight,
> language_model.model.model.layers.13.post_attention_layernorm.weight,
> language_model.model.model.layers.14.input_layernorm.weight,
> language_model.model.model.layers.14.linear_attn.A_log,
> language_model.model.model.layers.14.linear_attn.conv1d.weight,
> language_model.model.model.layers.14.linear_attn.dt_bias,
> language_model.model.model.layers.14.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.14.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.14.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.14.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.14.linear_attn.norm.weight,
> language_model.model.model.layers.14.linear_attn.out_proj.weight,
> language_model.model.model.layers.14.mlp.down_proj.biases,
> language_model.model.model.layers.14.mlp.down_proj.scales,
> language_model.model.model.layers.14.mlp.down_proj.weight,
> language_model.model.model.layers.14.mlp.gate_proj.biases,
> language_model.model.model.layers.14.mlp.gate_proj.scales,
> language_model.model.model.layers.14.mlp.gate_proj.weight,
> language_model.model.model.layers.14.mlp.up_proj.biases,
> language_model.model.model.layers.14.mlp.up_proj.scales,
> language_model.model.model.layers.14.mlp.up_proj.weight,
> language_model.model.model.layers.14.post_attention_layernorm.weight,
> language_model.model.model.layers.15.input_layernorm.weight,
> language_model.model.model.layers.15.mlp.down_proj.biases,
> language_model.model.model.layers.15.mlp.down_proj.scales,
> language_model.model.model.layers.15.mlp.down_proj.weight,
> language_model.model.model.layers.15.mlp.gate_proj.biases,
> language_model.model.model.layers.15.mlp.gate_proj.scales,
> language_model.model.model.layers.15.mlp.gate_proj.weight,
> language_model.model.model.layers.15.mlp.up_proj.biases,
> language_model.model.model.layers.15.mlp.up_proj.scales,
> language_model.model.model.layers.15.mlp.up_proj.weight,
> language_model.model.model.layers.15.post_attention_layernorm.weight,
> language_model.model.model.layers.15.self_attn.k_norm.weight,
> language_model.model.model.layers.15.self_attn.k_proj.biases,
> language_model.model.model.layers.15.self_attn.k_proj.scales,
> language_model.model.model.layers.15.self_attn.k_proj.weight,
> language_model.model.model.layers.15.self_attn.o_proj.biases,
> language_model.model.model.layers.15.self_attn.o_proj.scales,
> language_model.model.model.layers.15.self_attn.o_proj.weight,
> language_model.model.model.layers.15.self_attn.q_norm.weight,
> language_model.model.model.layers.15.self_attn.q_proj.biases,
> language_model.model.model.layers.15.self_attn.q_proj.scales,
> language_model.model.model.layers.15.self_attn.q_proj.weight,
> language_model.model.model.layers.15.self_attn.v_proj.biases,
> language_model.model.model.layers.15.self_attn.v_proj.scales,
> language_model.model.model.layers.15.self_attn.v_proj.weight,
> language_model.model.model.layers.16.input_layernorm.weight,
> language_model.model.model.layers.16.linear_attn.A_log,
> language_model.model.model.layers.16.linear_attn.conv1d.weight,
> language_model.model.model.layers.16.linear_attn.dt_bias,
> language_model.model.model.layers.16.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.16.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.16.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.16.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.16.linear_attn.norm.weight,
> language_model.model.model.layers.16.linear_attn.out_proj.weight,
> language_model.model.model.layers.16.mlp.down_proj.biases,
> language_model.model.model.layers.16.mlp.down_proj.scales,
> language_model.model.model.layers.16.mlp.down_proj.weight,
> language_model.model.model.layers.16.mlp.gate_proj.biases,
> language_model.model.model.layers.16.mlp.gate_proj.scales,
> language_model.model.model.layers.16.mlp.gate_proj.weight,
> language_model.model.model.layers.16.mlp.up_proj.biases,
> language_model.model.model.layers.16.mlp.up_proj.scales,
> language_model.model.model.layers.16.mlp.up_proj.weight,
> language_model.model.model.layers.16.post_attention_layernorm.weight,
> language_model.model.model.layers.17.input_layernorm.weight,
> language_model.model.model.layers.17.linear_attn.A_log,
> language_model.model.model.layers.17.linear_attn.conv1d.weight,
> language_model.model.model.layers.17.linear_attn.dt_bias,
> language_model.model.model.layers.17.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.17.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.17.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.17.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.17.linear_attn.norm.weight,
> language_model.model.model.layers.17.linear_attn.out_proj.weight,
> language_model.model.model.layers.17.mlp.down_proj.biases,
> language_model.model.model.layers.17.mlp.down_proj.scales,
> language_model.model.model.layers.17.mlp.down_proj.weight,
> language_model.model.model.layers.17.mlp.gate_proj.biases,
> language_model.model.model.layers.17.mlp.gate_proj.scales,
> language_model.model.model.layers.17.mlp.gate_proj.weight,
> language_model.model.model.layers.17.mlp.up_proj.biases,
> language_model.model.model.layers.17.mlp.up_proj.scales,
> language_model.model.model.layers.17.mlp.up_proj.weight,
> language_model.model.model.layers.17.post_attention_layernorm.weight,
> language_model.model.model.layers.18.input_layernorm.weight,
> language_model.model.model.layers.18.linear_attn.A_log,
> language_model.model.model.layers.18.linear_attn.conv1d.weight,
> language_model.model.model.layers.18.linear_attn.dt_bias,
> language_model.model.model.layers.18.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.18.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.18.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.18.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.18.linear_attn.norm.weight,
> language_model.model.model.layers.18.linear_attn.out_proj.weight,
> language_model.model.model.layers.18.mlp.down_proj.biases,
> language_model.model.model.layers.18.mlp.down_proj.scales,
> language_model.model.model.layers.18.mlp.down_proj.weight,
> language_model.model.model.layers.18.mlp.gate_proj.biases,
> language_model.model.model.layers.18.mlp.gate_proj.scales,
> language_model.model.model.layers.18.mlp.gate_proj.weight,
> language_model.model.model.layers.18.mlp.up_proj.biases,
> language_model.model.model.layers.18.mlp.up_proj.scales,
> language_model.model.model.layers.18.mlp.up_proj.weight,
> language_model.model.model.layers.18.post_attention_layernorm.weight,
> language_model.model.model.layers.19.input_layernorm.weight,
> language_model.model.model.layers.19.mlp.down_proj.biases,
> language_model.model.model.layers.19.mlp.down_proj.scales,
> language_model.model.model.layers.19.mlp.down_proj.weight,
> language_model.model.model.layers.19.mlp.gate_proj.biases,
> language_model.model.model.layers.19.mlp.gate_proj.scales,
> language_model.model.model.layers.19.mlp.gate_proj.weight,
> language_model.model.model.layers.19.mlp.up_proj.biases,
> language_model.model.model.layers.19.mlp.up_proj.scales,
> language_model.model.model.layers.19.mlp.up_proj.weight,
> language_model.model.model.layers.19.post_attention_layernorm.weight,
> language_model.model.model.layers.19.self_attn.k_norm.weight,
> language_model.model.model.layers.19.self_attn.k_proj.biases,
> language_model.model.model.layers.19.self_attn.k_proj.scales,
> language_model.model.model.layers.19.self_attn.k_proj.weight,
> language_model.model.model.layers.19.self_attn.o_proj.biases,
> language_model.model.model.layers.19.self_attn.o_proj.scales,
> language_model.model.model.layers.19.self_attn.o_proj.weight,
> language_model.model.model.layers.19.self_attn.q_norm.weight,
> language_model.model.model.layers.19.self_attn.q_proj.biases,
> language_model.model.model.layers.19.self_attn.q_proj.scales,
> language_model.model.model.layers.19.self_attn.q_proj.weight,
> language_model.model.model.layers.19.self_attn.v_proj.biases,
> language_model.model.model.layers.19.self_attn.v_proj.scales,
> language_model.model.model.layers.19.self_attn.v_proj.weight,
> language_model.model.model.layers.2.input_layernorm.weight,
> language_model.model.model.layers.2.linear_attn.A_log,
> language_model.model.model.layers.2.linear_attn.conv1d.weight,
> language_model.model.model.layers.2.linear_attn.dt_bias,
> language_model.model.model.layers.2.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.2.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.2.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.2.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.2.linear_attn.norm.weight,
> language_model.model.model.layers.2.linear_attn.out_proj.weight,
> language_model.model.model.layers.2.mlp.down_proj.biases,
> language_model.model.model.layers.2.mlp.down_proj.scales,
> language_model.model.model.layers.2.mlp.down_proj.weight,
> language_model.model.model.layers.2.mlp.gate_proj.biases,
> language_model.model.model.layers.2.mlp.gate_proj.scales,
> language_model.model.model.layers.2.mlp.gate_proj.weight,
> language_model.model.model.layers.2.mlp.up_proj.biases,
> language_model.model.model.layers.2.mlp.up_proj.scales,
> language_model.model.model.layers.2.mlp.up_proj.weight,
> language_model.model.model.layers.2.post_attention_layernorm.weight,
> language_model.model.model.layers.20.input_layernorm.weight,
> language_model.model.model.layers.20.linear_attn.A_log,
> language_model.model.model.layers.20.linear_attn.conv1d.weight,
> language_model.model.model.layers.20.linear_attn.dt_bias,
> language_model.model.model.layers.20.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.20.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.20.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.20.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.20.linear_attn.norm.weight,
> language_model.model.model.layers.20.linear_attn.out_proj.weight,
> language_model.model.model.layers.20.mlp.down_proj.biases,
> language_model.model.model.layers.20.mlp.down_proj.scales,
> language_model.model.model.layers.20.mlp.down_proj.weight,
> language_model.model.model.layers.20.mlp.gate_proj.biases,
> language_model.model.model.layers.20.mlp.gate_proj.scales,
> language_model.model.model.layers.20.mlp.gate_proj.weight,
> language_model.model.model.layers.20.mlp.up_proj.biases,
> language_model.model.model.layers.20.mlp.up_proj.scales,
> language_model.model.model.layers.20.mlp.up_proj.weight,
> language_model.model.model.layers.20.post_attention_layernorm.weight,
> language_model.model.model.layers.21.input_layernorm.weight,
> language_model.model.model.layers.21.linear_attn.A_log,
> language_model.model.model.layers.21.linear_attn.conv1d.weight,
> language_model.model.model.layers.21.linear_attn.dt_bias,
> language_model.model.model.layers.21.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.21.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.21.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.21.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.21.linear_attn.norm.weight,
> language_model.model.model.layers.21.linear_attn.out_proj.weight,
> language_model.model.model.layers.21.mlp.down_proj.biases,
> language_model.model.model.layers.21.mlp.down_proj.scales,
> language_model.model.model.layers.21.mlp.down_proj.weight,
> language_model.model.model.layers.21.mlp.gate_proj.biases,
> language_model.model.model.layers.21.mlp.gate_proj.scales,
> language_model.model.model.layers.21.mlp.gate_proj.weight,
> language_model.model.model.layers.21.mlp.up_proj.biases,
> language_model.model.model.layers.21.mlp.up_proj.scales,
> language_model.model.model.layers.21.mlp.up_proj.weight,
> language_model.model.model.layers.21.post_attention_layernorm.weight,
> language_model.model.model.layers.22.input_layernorm.weight,
> language_model.model.model.layers.22.linear_attn.A_log,
> language_model.model.model.layers.22.linear_attn.conv1d.weight,
> language_model.model.model.layers.22.linear_attn.dt_bias,
> language_model.model.model.layers.22.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.22.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.22.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.22.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.22.linear_attn.norm.weight,
> language_model.model.model.layers.22.linear_attn.out_proj.weight,
> language_model.model.model.layers.22.mlp.down_proj.biases,
> language_model.model.model.layers.22.mlp.down_proj.scales,
> language_model.model.model.layers.22.mlp.down_proj.weight,
> language_model.model.model.layers.22.mlp.gate_proj.biases,
> language_model.model.model.layers.22.mlp.gate_proj.scales,
> language_model.model.model.layers.22.mlp.gate_proj.weight,
> language_model.model.model.layers.22.mlp.up_proj.biases,
> language_model.model.model.layers.22.mlp.up_proj.scales,
> language_model.model.model.layers.22.mlp.up_proj.weight,
> language_model.model.model.layers.22.post_attention_layernorm.weight,
> language_model.model.model.layers.23.input_layernorm.weight,
> language_model.model.model.layers.23.mlp.down_proj.biases,
> language_model.model.model.layers.23.mlp.down_proj.scales,
> language_model.model.model.layers.23.mlp.down_proj.weight,
> language_model.model.model.layers.23.mlp.gate_proj.biases,
> language_model.model.model.layers.23.mlp.gate_proj.scales,
> language_model.model.model.layers.23.mlp.gate_proj.weight,
> language_model.model.model.layers.23.mlp.up_proj.biases,
> language_model.model.model.layers.23.mlp.up_proj.scales,
> language_model.model.model.layers.23.mlp.up_proj.weight,
> language_model.model.model.layers.23.post_attention_layernorm.weight,
> language_model.model.model.layers.23.self_attn.k_norm.weight,
> language_model.model.model.layers.23.self_attn.k_proj.biases,
> language_model.model.model.layers.23.self_attn.k_proj.scales,
> language_model.model.model.layers.23.self_attn.k_proj.weight,
> language_model.model.model.layers.23.self_attn.o_proj.biases,
> language_model.model.model.layers.23.self_attn.o_proj.scales,
> language_model.model.model.layers.23.self_attn.o_proj.weight,
> language_model.model.model.layers.23.self_attn.q_norm.weight,
> language_model.model.model.layers.23.self_attn.q_proj.biases,
> language_model.model.model.layers.23.self_attn.q_proj.scales,
> language_model.model.model.layers.23.self_attn.q_proj.weight,
> language_model.model.model.layers.23.self_attn.v_proj.biases,
> language_model.model.model.layers.23.self_attn.v_proj.scales,
> language_model.model.model.layers.23.self_attn.v_proj.weight,
> language_model.model.model.layers.3.input_layernorm.weight,
> language_model.model.model.layers.3.mlp.down_proj.biases,
> language_model.model.model.layers.3.mlp.down_proj.scales,
> language_model.model.model.layers.3.mlp.down_proj.weight,
> language_model.model.model.layers.3.mlp.gate_proj.biases,
> language_model.model.model.layers.3.mlp.gate_proj.scales,
> language_model.model.model.layers.3.mlp.gate_proj.weight,
> language_model.model.model.layers.3.mlp.up_proj.biases,
> language_model.model.model.layers.3.mlp.up_proj.scales,
> language_model.model.model.layers.3.mlp.up_proj.weight,
> language_model.model.model.layers.3.post_attention_layernorm.weight,
> language_model.model.model.layers.3.self_attn.k_norm.weight,
> language_model.model.model.layers.3.self_attn.k_proj.biases,
> language_model.model.model.layers.3.self_attn.k_proj.scales,
> language_model.model.model.layers.3.self_attn.k_proj.weight,
> language_model.model.model.layers.3.self_attn.o_proj.biases,
> language_model.model.model.layers.3.self_attn.o_proj.scales,
> language_model.model.model.layers.3.self_attn.o_proj.weight,
> language_model.model.model.layers.3.self_attn.q_norm.weight,
> language_model.model.model.layers.3.self_attn.q_proj.biases,
> language_model.model.model.layers.3.self_attn.q_proj.scales,
> language_model.model.model.layers.3.self_attn.q_proj.weight,
> language_model.model.model.layers.3.self_attn.v_proj.biases,
> language_model.model.model.layers.3.self_attn.v_proj.scales,
> language_model.model.model.layers.3.self_attn.v_proj.weight,
> language_model.model.model.layers.4.input_layernorm.weight,
> language_model.model.model.layers.4.linear_attn.A_log,
> language_model.model.model.layers.4.linear_attn.conv1d.weight,
> language_model.model.model.layers.4.linear_attn.dt_bias,
> language_model.model.model.layers.4.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.4.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.4.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.4.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.4.linear_attn.norm.weight,
> language_model.model.model.layers.4.linear_attn.out_proj.weight,
> language_model.model.model.layers.4.mlp.down_proj.biases,
> language_model.model.model.layers.4.mlp.down_proj.scales,
> language_model.model.model.layers.4.mlp.down_proj.weight,
> language_model.model.model.layers.4.mlp.gate_proj.biases,
> language_model.model.model.layers.4.mlp.gate_proj.scales,
> language_model.model.model.layers.4.mlp.gate_proj.weight,
> language_model.model.model.layers.4.mlp.up_proj.biases,
> language_model.model.model.layers.4.mlp.up_proj.scales,
> language_model.model.model.layers.4.mlp.up_proj.weight,
> language_model.model.model.layers.4.post_attention_layernorm.weight,
> language_model.model.model.layers.5.input_layernorm.weight,
> language_model.model.model.layers.5.linear_attn.A_log,
> language_model.model.model.layers.5.linear_attn.conv1d.weight,
> language_model.model.model.layers.5.linear_attn.dt_bias,
> language_model.model.model.layers.5.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.5.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.5.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.5.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.5.linear_attn.norm.weight,
> language_model.model.model.layers.5.linear_attn.out_proj.weight,
> language_model.model.model.layers.5.mlp.down_proj.biases,
> language_model.model.model.layers.5.mlp.down_proj.scales,
> language_model.model.model.layers.5.mlp.down_proj.weight,
> language_model.model.model.layers.5.mlp.gate_proj.biases,
> language_model.model.model.layers.5.mlp.gate_proj.scales,
> language_model.model.model.layers.5.mlp.gate_proj.weight,
> language_model.model.model.layers.5.mlp.up_proj.biases,
> language_model.model.model.layers.5.mlp.up_proj.scales,
> language_model.model.model.layers.5.mlp.up_proj.weight,
> language_model.model.model.layers.5.post_attention_layernorm.weight,
> language_model.model.model.layers.6.input_layernorm.weight,
> language_model.model.model.layers.6.linear_attn.A_log,
> language_model.model.model.layers.6.linear_attn.conv1d.weight,
> language_model.model.model.layers.6.linear_attn.dt_bias,
> language_model.model.model.layers.6.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.6.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.6.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.6.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.6.linear_attn.norm.weight,
> language_model.model.model.layers.6.linear_attn.out_proj.weight,
> language_model.model.model.layers.6.mlp.down_proj.biases,
> language_model.model.model.layers.6.mlp.down_proj.scales,
> language_model.model.model.layers.6.mlp.down_proj.weight,
> language_model.model.model.layers.6.mlp.gate_proj.biases,
> language_model.model.model.layers.6.mlp.gate_proj.scales,
> language_model.model.model.layers.6.mlp.gate_proj.weight,
> language_model.model.model.layers.6.mlp.up_proj.biases,
> language_model.model.model.layers.6.mlp.up_proj.scales,
> language_model.model.model.layers.6.mlp.up_proj.weight,
> language_model.model.model.layers.6.post_attention_layernorm.weight,
> language_model.model.model.layers.7.input_layernorm.weight,
> language_model.model.model.layers.7.mlp.down_proj.biases,
> language_model.model.model.layers.7.mlp.down_proj.scales,
> language_model.model.model.layers.7.mlp.down_proj.weight,
> language_model.model.model.layers.7.mlp.gate_proj.biases,
> language_model.model.model.layers.7.mlp.gate_proj.scales,
> language_model.model.model.layers.7.mlp.gate_proj.weight,
> language_model.model.model.layers.7.mlp.up_proj.biases,
> language_model.model.model.layers.7.mlp.up_proj.scales,
> language_model.model.model.layers.7.mlp.up_proj.weight,
> language_model.model.model.layers.7.post_attention_layernorm.weight,
> language_model.model.model.layers.7.self_attn.k_norm.weight,
> language_model.model.model.layers.7.self_attn.k_proj.biases,
> language_model.model.model.layers.7.self_attn.k_proj.scales,
> language_model.model.model.layers.7.self_attn.k_proj.weight,
> language_model.model.model.layers.7.self_attn.o_proj.biases,
> language_model.model.model.layers.7.self_attn.o_proj.scales,
> language_model.model.model.layers.7.self_attn.o_proj.weight,
> language_model.model.model.layers.7.self_attn.q_norm.weight,
> language_model.model.model.layers.7.self_attn.q_proj.biases,
> language_model.model.model.layers.7.self_attn.q_proj.scales,
> language_model.model.model.layers.7.self_attn.q_proj.weight,
> language_model.model.model.layers.7.self_attn.v_proj.biases,
> language_model.model.model.layers.7.self_attn.v_proj.scales,
> language_model.model.model.layers.7.self_attn.v_proj.weight,
> language_model.model.model.layers.8.input_layernorm.weight,
> language_model.model.model.layers.8.linear_attn.A_log,
> language_model.model.model.layers.8.linear_attn.conv1d.weight,
> language_model.model.model.layers.8.linear_attn.dt_bias,
> language_model.model.model.layers.8.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.8.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.8.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.8.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.8.linear_attn.norm.weight,
> language_model.model.model.layers.8.linear_attn.out_proj.weight,
> language_model.model.model.layers.8.mlp.down_proj.biases,
> language_model.model.model.layers.8.mlp.down_proj.scales,
> language_model.model.model.layers.8.mlp.down_proj.weight,
> language_model.model.model.layers.8.mlp.gate_proj.biases,
> language_model.model.model.layers.8.mlp.gate_proj.scales,
> language_model.model.model.layers.8.mlp.gate_proj.weight,
> language_model.model.model.layers.8.mlp.up_proj.biases,
> language_model.model.model.layers.8.mlp.up_proj.scales,
> language_model.model.model.layers.8.mlp.up_proj.weight,
> language_model.model.model.layers.8.post_attention_layernorm.weight,
> language_model.model.model.layers.9.input_layernorm.weight,
> language_model.model.model.layers.9.linear_attn.A_log,
> language_model.model.model.layers.9.linear_attn.conv1d.weight,
> language_model.model.model.layers.9.linear_attn.dt_bias,
> language_model.model.model.layers.9.linear_attn.in_proj_a.weight,
> language_model.model.model.layers.9.linear_attn.in_proj_b.weight,
> language_model.model.model.layers.9.linear_attn.in_proj_qkv.weight,
> language_model.model.model.layers.9.linear_attn.in_proj_z.weight,
> language_model.model.model.layers.9.linear_attn.norm.weight,
> language_model.model.model.layers.9.linear_attn.out_proj.weight,
> language_model.model.model.layers.9.mlp.down_proj.biases,
> language_model.model.model.layers.9.mlp.down_proj.scales,
> language_model.model.model.layers.9.mlp.down_proj.weight,
> language_model.model.model.layers.9.mlp.gate_proj.biases,
> language_model.model.model.layers.9.mlp.gate_proj.scales,
> language_model.model.model.layers.9.mlp.gate_proj.weight,
> language_model.model.model.layers.9.mlp.up_proj.biases,
> language_model.model.model.layers.9.mlp.up_proj.scales,
> language_model.model.model.layers.9.mlp.up_proj.weight,
> language_model.model.model.layers.9.post_attention_layernorm.weight,
> language_model.model.model.norm.weight.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20124, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19515, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 941, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 788, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 664, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 512 parameters not in model:
language_model.model.model.embed_tokens.weight,
language_model.model.model.layers.0.input_layernorm.weight,
language_model.model.model.layers.0.linear_attn.A_log,
language_model.model.model.layers.0.linear_attn.conv1d.weight,
language_model.model.model.layers.0.linear_attn.dt_bias,
language_model.model.model.layers.0.linear_attn.in_proj_a.weight,
language_model.model.model.layers.0.linear_attn.in_proj_b.weight,
language_model.model.model.layers.0.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.0.linear_attn.in_proj_z.weight,
language_model.model.model.layers.0.linear_attn.norm.weight,
language_model.model.model.layers.0.linear_attn.out_proj.weight,
language_model.model.model.layers.0.mlp.down_proj.biases,
language_model.model.model.layers.0.mlp.down_proj.scales,
language_model.model.model.layers.0.mlp.down_proj.weight,
language_model.model.model.layers.0.mlp.gate_proj.biases,
language_model.model.model.layers.0.mlp.gate_proj.scales,
language_model.model.model.layers.0.mlp.gate_proj.weight,
language_model.model.model.layers.0.mlp.up_proj.biases,
language_model.model.model.layers.0.mlp.up_proj.scales,
language_model.model.model.layers.0.mlp.up_proj.weight,
language_model.model.model.layers.0.post_attention_layernorm.weight,
language_model.model.model.layers.1.input_layernorm.weight,
language_model.model.model.layers.1.linear_attn.A_log,
language_model.model.model.layers.1.linear_attn.conv1d.weight,
language_model.model.model.layers.1.linear_attn.dt_bias,
language_model.model.model.layers.1.linear_attn.in_proj_a.weight,
language_model.model.model.layers.1.linear_attn.in_proj_b.weight,
language_model.model.model.layers.1.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.1.linear_attn.in_proj_z.weight,
language_model.model.model.layers.1.linear_attn.norm.weight,
language_model.model.model.layers.1.linear_attn.out_proj.weight,
language_model.model.model.layers.1.mlp.down_proj.biases,
language_model.model.model.layers.1.mlp.down_proj.scales,
language_model.model.model.layers.1.mlp.down_proj.weight,
language_model.model.model.layers.1.mlp.gate_proj.biases,
language_model.model.model.layers.1.mlp.gate_proj.scales,
language_model.model.model.layers.1.mlp.gate_proj.weight,
language_model.model.model.layers.1.mlp.up_proj.biases,
language_model.model.model.layers.1.mlp.up_proj.scales,
language_model.model.model.layers.1.mlp.up_proj.weight,
language_model.model.model.layers.1.post_attention_layernorm.weight,
language_model.model.model.layers.10.input_layernorm.weight,
language_model.model.model.layers.10.linear_attn.A_log,
language_model.model.model.layers.10.linear_attn.conv1d.weight,
language_model.model.model.layers.10.linear_attn.dt_bias,
language_model.model.model.layers.10.linear_attn.in_proj_a.weight,
language_model.model.model.layers.10.linear_attn.in_proj_b.weight,
language_model.model.model.layers.10.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.10.linear_attn.in_proj_z.weight,
language_model.model.model.layers.10.linear_attn.norm.weight,
language_model.model.model.layers.10.linear_attn.out_proj.weight,
language_model.model.model.layers.10.mlp.down_proj.biases,
language_model.model.model.layers.10.mlp.down_proj.scales,
language_model.model.model.layers.10.mlp.down_proj.weight,
language_model.model.model.layers.10.mlp.gate_proj.biases,
language_model.model.model.layers.10.mlp.gate_proj.scales,
language_model.model.model.layers.10.mlp.gate_proj.weight,
language_model.model.model.layers.10.mlp.up_proj.biases,
language_model.model.model.layers.10.mlp.up_proj.scales,
language_model.model.model.layers.10.mlp.up_proj.weight,
language_model.model.model.layers.10.post_attention_layernorm.weight,
language_model.model.model.layers.11.input_layernorm.weight,
language_model.model.model.layers.11.mlp.down_proj.biases,
language_model.model.model.layers.11.mlp.down_proj.scales,
language_model.model.model.layers.11.mlp.down_proj.weight,
language_model.model.model.layers.11.mlp.gate_proj.biases,
language_model.model.model.layers.11.mlp.gate_proj.scales,
language_model.model.model.layers.11.mlp.gate_proj.weight,
language_model.model.model.layers.11.mlp.up_proj.biases,
language_model.model.model.layers.11.mlp.up_proj.scales,
language_model.model.model.layers.11.mlp.up_proj.weight,
language_model.model.model.layers.11.post_attention_layernorm.weight,
language_model.model.model.layers.11.self_attn.k_norm.weight,
language_model.model.model.layers.11.self_attn.k_proj.biases,
language_model.model.model.layers.11.self_attn.k_proj.scales,
language_model.model.model.layers.11.self_attn.k_proj.weight,
language_model.model.model.layers.11.self_attn.o_proj.biases,
language_model.model.model.layers.11.self_attn.o_proj.scales,
language_model.model.model.layers.11.self_attn.o_proj.weight,
language_model.model.model.layers.11.self_attn.q_norm.weight,
language_model.model.model.layers.11.self_attn.q_proj.biases,
language_model.model.model.layers.11.self_attn.q_proj.scales,
language_model.model.model.layers.11.self_attn.q_proj.weight,
language_model.model.model.layers.11.self_attn.v_proj.biases,
language_model.model.model.layers.11.self_attn.v_proj.scales,
language_model.model.model.layers.11.self_attn.v_proj.weight,
language_model.model.model.layers.12.input_layernorm.weight,
language_model.model.model.layers.12.linear_attn.A_log,
language_model.model.model.layers.12.linear_attn.conv1d.weight,
language_model.model.model.layers.12.linear_attn.dt_bias,
language_model.model.model.layers.12.linear_attn.in_proj_a.weight,
language_model.model.model.layers.12.linear_attn.in_proj_b.weight,
language_model.model.model.layers.12.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.12.linear_attn.in_proj_z.weight,
language_model.model.model.layers.12.linear_attn.norm.weight,
language_model.model.model.layers.12.linear_attn.out_proj.weight,
language_model.model.model.layers.12.mlp.down_proj.biases,
language_model.model.model.layers.12.mlp.down_proj.scales,
language_model.model.model.layers.12.mlp.down_proj.weight,
language_model.model.model.layers.12.mlp.gate_proj.biases,
language_model.model.model.layers.12.mlp.gate_proj.scales,
language_model.model.model.layers.12.mlp.gate_proj.weight,
language_model.model.model.layers.12.mlp.up_proj.biases,
language_model.model.model.layers.12.mlp.up_proj.scales,
language_model.model.model.layers.12.mlp.up_proj.weight,
language_model.model.model.layers.12.post_attention_layernorm.weight,
language_model.model.model.layers.13.input_layernorm.weight,
language_model.model.model.layers.13.linear_attn.A_log,
language_model.model.model.layers.13.linear_attn.conv1d.weight,
language_model.model.model.layers.13.linear_attn.dt_bias,
language_model.model.model.layers.13.linear_attn.in_proj_a.weight,
language_model.model.model.layers.13.linear_attn.in_proj_b.weight,
language_model.model.model.layers.13.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.13.linear_attn.in_proj_z.weight,
language_model.model.model.layers.13.linear_attn.norm.weight,
language_model.model.model.layers.13.linear_attn.out_proj.weight,
language_model.model.model.layers.13.mlp.down_proj.biases,
language_model.model.model.layers.13.mlp.down_proj.scales,
language_model.model.model.layers.13.mlp.down_proj.weight,
language_model.model.model.layers.13.mlp.gate_proj.biases,
language_model.model.model.layers.13.mlp.gate_proj.scales,
language_model.model.model.layers.13.mlp.gate_proj.weight,
language_model.model.model.layers.13.mlp.up_proj.biases,
language_model.model.model.layers.13.mlp.up_proj.scales,
language_model.model.model.layers.13.mlp.up_proj.weight,
language_model.model.model.layers.13.post_attention_layernorm.weight,
language_model.model.model.layers.14.input_layernorm.weight,
language_model.model.model.layers.14.linear_attn.A_log,
language_model.model.model.layers.14.linear_attn.conv1d.weight,
language_model.model.model.layers.14.linear_attn.dt_bias,
language_model.model.model.layers.14.linear_attn.in_proj_a.weight,
language_model.model.model.layers.14.linear_attn.in_proj_b.weight,
language_model.model.model.layers.14.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.14.linear_attn.in_proj_z.weight,
language_model.model.model.layers.14.linear_attn.norm.weight,
language_model.model.model.layers.14.linear_attn.out_proj.weight,
language_model.model.model.layers.14.mlp.down_proj.biases,
language_model.model.model.layers.14.mlp.down_proj.scales,
language_model.model.model.layers.14.mlp.down_proj.weight,
language_model.model.model.layers.14.mlp.gate_proj.biases,
language_model.model.model.layers.14.mlp.gate_proj.scales,
language_model.model.model.layers.14.mlp.gate_proj.weight,
language_model.model.model.layers.14.mlp.up_proj.biases,
language_model.model.model.layers.14.mlp.up_proj.scales,
language_model.model.model.layers.14.mlp.up_proj.weight,
language_model.model.model.layers.14.post_attention_layernorm.weight,
language_model.model.model.layers.15.input_layernorm.weight,
language_model.model.model.layers.15.mlp.down_proj.biases,
language_model.model.model.layers.15.mlp.down_proj.scales,
language_model.model.model.layers.15.mlp.down_proj.weight,
language_model.model.model.layers.15.mlp.gate_proj.biases,
language_model.model.model.layers.15.mlp.gate_proj.scales,
language_model.model.model.layers.15.mlp.gate_proj.weight,
language_model.model.model.layers.15.mlp.up_proj.biases,
language_model.model.model.layers.15.mlp.up_proj.scales,
language_model.model.model.layers.15.mlp.up_proj.weight,
language_model.model.model.layers.15.post_attention_layernorm.weight,
language_model.model.model.layers.15.self_attn.k_norm.weight,
language_model.model.model.layers.15.self_attn.k_proj.biases,
language_model.model.model.layers.15.self_attn.k_proj.scales,
language_model.model.model.layers.15.self_attn.k_proj.weight,
language_model.model.model.layers.15.self_attn.o_proj.biases,
language_model.model.model.layers.15.self_attn.o_proj.scales,
language_model.model.model.layers.15.self_attn.o_proj.weight,
language_model.model.model.layers.15.self_attn.q_norm.weight,
language_model.model.model.layers.15.self_attn.q_proj.biases,
language_model.model.model.layers.15.self_attn.q_proj.scales,
language_model.model.model.layers.15.self_attn.q_proj.weight,
language_model.model.model.layers.15.self_attn.v_proj.biases,
language_model.model.model.layers.15.self_attn.v_proj.scales,
language_model.model.model.layers.15.self_attn.v_proj.weight,
language_model.model.model.layers.16.input_layernorm.weight,
language_model.model.model.layers.16.linear_attn.A_log,
language_model.model.model.layers.16.linear_attn.conv1d.weight,
language_model.model.model.layers.16.linear_attn.dt_bias,
language_model.model.model.layers.16.linear_attn.in_proj_a.weight,
language_model.model.model.layers.16.linear_attn.in_proj_b.weight,
language_model.model.model.layers.16.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.16.linear_attn.in_proj_z.weight,
language_model.model.model.layers.16.linear_attn.norm.weight,
language_model.model.model.layers.16.linear_attn.out_proj.weight,
language_model.model.model.layers.16.mlp.down_proj.biases,
language_model.model.model.layers.16.mlp.down_proj.scales,
language_model.model.model.layers.16.mlp.down_proj.weight,
language_model.model.model.layers.16.mlp.gate_proj.biases,
language_model.model.model.layers.16.mlp.gate_proj.scales,
language_model.model.model.layers.16.mlp.gate_proj.weight,
language_model.model.model.layers.16.mlp.up_proj.biases,
language_model.model.model.layers.16.mlp.up_proj.scales,
language_model.model.model.layers.16.mlp.up_proj.weight,
language_model.model.model.layers.16.post_attention_layernorm.weight,
language_model.model.model.layers.17.input_layernorm.weight,
language_model.model.model.layers.17.linear_attn.A_log,
language_model.model.model.layers.17.linear_attn.conv1d.weight,
language_model.model.model.layers.17.linear_attn.dt_bias,
language_model.model.model.layers.17.linear_attn.in_proj_a.weight,
language_model.model.model.layers.17.linear_attn.in_proj_b.weight,
language_model.model.model.layers.17.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.17.linear_attn.in_proj_z.weight,
language_model.model.model.layers.17.linear_attn.norm.weight,
language_model.model.model.layers.17.linear_attn.out_proj.weight,
language_model.model.model.layers.17.mlp.down_proj.biases,
language_model.model.model.layers.17.mlp.down_proj.scales,
language_model.model.model.layers.17.mlp.down_proj.weight,
language_model.model.model.layers.17.mlp.gate_proj.biases,
language_model.model.model.layers.17.mlp.gate_proj.scales,
language_model.model.model.layers.17.mlp.gate_proj.weight,
language_model.model.model.layers.17.mlp.up_proj.biases,
language_model.model.model.layers.17.mlp.up_proj.scales,
language_model.model.model.layers.17.mlp.up_proj.weight,
language_model.model.model.layers.17.post_attention_layernorm.weight,
language_model.model.model.layers.18.input_layernorm.weight,
language_model.model.model.layers.18.linear_attn.A_log,
language_model.model.model.layers.18.linear_attn.conv1d.weight,
language_model.model.model.layers.18.linear_attn.dt_bias,
language_model.model.model.layers.18.linear_attn.in_proj_a.weight,
language_model.model.model.layers.18.linear_attn.in_proj_b.weight,
language_model.model.model.layers.18.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.18.linear_attn.in_proj_z.weight,
language_model.model.model.layers.18.linear_attn.norm.weight,
language_model.model.model.layers.18.linear_attn.out_proj.weight,
language_model.model.model.layers.18.mlp.down_proj.biases,
language_model.model.model.layers.18.mlp.down_proj.scales,
language_model.model.model.layers.18.mlp.down_proj.weight,
language_model.model.model.layers.18.mlp.gate_proj.biases,
language_model.model.model.layers.18.mlp.gate_proj.scales,
language_model.model.model.layers.18.mlp.gate_proj.weight,
language_model.model.model.layers.18.mlp.up_proj.biases,
language_model.model.model.layers.18.mlp.up_proj.scales,
language_model.model.model.layers.18.mlp.up_proj.weight,
language_model.model.model.layers.18.post_attention_layernorm.weight,
language_model.model.model.layers.19.input_layernorm.weight,
language_model.model.model.layers.19.mlp.down_proj.biases,
language_model.model.model.layers.19.mlp.down_proj.scales,
language_model.model.model.layers.19.mlp.down_proj.weight,
language_model.model.model.layers.19.mlp.gate_proj.biases,
language_model.model.model.layers.19.mlp.gate_proj.scales,
language_model.model.model.layers.19.mlp.gate_proj.weight,
language_model.model.model.layers.19.mlp.up_proj.biases,
language_model.model.model.layers.19.mlp.up_proj.scales,
language_model.model.model.layers.19.mlp.up_proj.weight,
language_model.model.model.layers.19.post_attention_layernorm.weight,
language_model.model.model.layers.19.self_attn.k_norm.weight,
language_model.model.model.layers.19.self_attn.k_proj.biases,
language_model.model.model.layers.19.self_attn.k_proj.scales,
language_model.model.model.layers.19.self_attn.k_proj.weight,
language_model.model.model.layers.19.self_attn.o_proj.biases,
language_model.model.model.layers.19.self_attn.o_proj.scales,
language_model.model.model.layers.19.self_attn.o_proj.weight,
language_model.model.model.layers.19.self_attn.q_norm.weight,
language_model.model.model.layers.19.self_attn.q_proj.biases,
language_model.model.model.layers.19.self_attn.q_proj.scales,
language_model.model.model.layers.19.self_attn.q_proj.weight,
language_model.model.model.layers.19.self_attn.v_proj.biases,
language_model.model.model.layers.19.self_attn.v_proj.scales,
language_model.model.model.layers.19.self_attn.v_proj.weight,
language_model.model.model.layers.2.input_layernorm.weight,
language_model.model.model.layers.2.linear_attn.A_log,
language_model.model.model.layers.2.linear_attn.conv1d.weight,
language_model.model.model.layers.2.linear_attn.dt_bias,
language_model.model.model.layers.2.linear_attn.in_proj_a.weight,
language_model.model.model.layers.2.linear_attn.in_proj_b.weight,
language_model.model.model.layers.2.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.2.linear_attn.in_proj_z.weight,
language_model.model.model.layers.2.linear_attn.norm.weight,
language_model.model.model.layers.2.linear_attn.out_proj.weight,
language_model.model.model.layers.2.mlp.down_proj.biases,
language_model.model.model.layers.2.mlp.down_proj.scales,
language_model.model.model.layers.2.mlp.down_proj.weight,
language_model.model.model.layers.2.mlp.gate_proj.biases,
language_model.model.model.layers.2.mlp.gate_proj.scales,
language_model.model.model.layers.2.mlp.gate_proj.weight,
language_model.model.model.layers.2.mlp.up_proj.biases,
language_model.model.model.layers.2.mlp.up_proj.scales,
language_model.model.model.layers.2.mlp.up_proj.weight,
language_model.model.model.layers.2.post_attention_layernorm.weight,
language_model.model.model.layers.20.input_layernorm.weight,
language_model.model.model.layers.20.linear_attn.A_log,
language_model.model.model.layers.20.linear_attn.conv1d.weight,
language_model.model.model.layers.20.linear_attn.dt_bias,
language_model.model.model.layers.20.linear_attn.in_proj_a.weight,
language_model.model.model.layers.20.linear_attn.in_proj_b.weight,
language_model.model.model.layers.20.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.20.linear_attn.in_proj_z.weight,
language_model.model.model.layers.20.linear_attn.norm.weight,
language_model.model.model.layers.20.linear_attn.out_proj.weight,
language_model.model.model.layers.20.mlp.down_proj.biases,
language_model.model.model.layers.20.mlp.down_proj.scales,
language_model.model.model.layers.20.mlp.down_proj.weight,
language_model.model.model.layers.20.mlp.gate_proj.biases,
language_model.model.model.layers.20.mlp.gate_proj.scales,
language_model.model.model.layers.20.mlp.gate_proj.weight,
language_model.model.model.layers.20.mlp.up_proj.biases,
language_model.model.model.layers.20.mlp.up_proj.scales,
language_model.model.model.layers.20.mlp.up_proj.weight,
language_model.model.model.layers.20.post_attention_layernorm.weight,
language_model.model.model.layers.21.input_layernorm.weight,
language_model.model.model.layers.21.linear_attn.A_log,
language_model.model.model.layers.21.linear_attn.conv1d.weight,
language_model.model.model.layers.21.linear_attn.dt_bias,
language_model.model.model.layers.21.linear_attn.in_proj_a.weight,
language_model.model.model.layers.21.linear_attn.in_proj_b.weight,
language_model.model.model.layers.21.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.21.linear_attn.in_proj_z.weight,
language_model.model.model.layers.21.linear_attn.norm.weight,
language_model.model.model.layers.21.linear_attn.out_proj.weight,
language_model.model.model.layers.21.mlp.down_proj.biases,
language_model.model.model.layers.21.mlp.down_proj.scales,
language_model.model.model.layers.21.mlp.down_proj.weight,
language_model.model.model.layers.21.mlp.gate_proj.biases,
language_model.model.model.layers.21.mlp.gate_proj.scales,
language_model.model.model.layers.21.mlp.gate_proj.weight,
language_model.model.model.layers.21.mlp.up_proj.biases,
language_model.model.model.layers.21.mlp.up_proj.scales,
language_model.model.model.layers.21.mlp.up_proj.weight,
language_model.model.model.layers.21.post_attention_layernorm.weight,
language_model.model.model.layers.22.input_layernorm.weight,
language_model.model.model.layers.22.linear_attn.A_log,
language_model.model.model.layers.22.linear_attn.conv1d.weight,
language_model.model.model.layers.22.linear_attn.dt_bias,
language_model.model.model.layers.22.linear_attn.in_proj_a.weight,
language_model.model.model.layers.22.linear_attn.in_proj_b.weight,
language_model.model.model.layers.22.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.22.linear_attn.in_proj_z.weight,
language_model.model.model.layers.22.linear_attn.norm.weight,
language_model.model.model.layers.22.linear_attn.out_proj.weight,
language_model.model.model.layers.22.mlp.down_proj.biases,
language_model.model.model.layers.22.mlp.down_proj.scales,
language_model.model.model.layers.22.mlp.down_proj.weight,
language_model.model.model.layers.22.mlp.gate_proj.biases,
language_model.model.model.layers.22.mlp.gate_proj.scales,
language_model.model.model.layers.22.mlp.gate_proj.weight,
language_model.model.model.layers.22.mlp.up_proj.biases,
language_model.model.model.layers.22.mlp.up_proj.scales,
language_model.model.model.layers.22.mlp.up_proj.weight,
language_model.model.model.layers.22.post_attention_layernorm.weight,
language_model.model.model.layers.23.input_layernorm.weight,
language_model.model.model.layers.23.mlp.down_proj.biases,
language_model.model.model.layers.23.mlp.down_proj.scales,
language_model.model.model.layers.23.mlp.down_proj.weight,
language_model.model.model.layers.23.mlp.gate_proj.biases,
language_model.model.model.layers.23.mlp.gate_proj.scales,
language_model.model.model.layers.23.mlp.gate_proj.weight,
language_model.model.model.layers.23.mlp.up_proj.biases,
language_model.model.model.layers.23.mlp.up_proj.scales,
language_model.model.model.layers.23.mlp.up_proj.weight,
language_model.model.model.layers.23.post_attention_layernorm.weight,
language_model.model.model.layers.23.self_attn.k_norm.weight,
language_model.model.model.layers.23.self_attn.k_proj.biases,
language_model.model.model.layers.23.self_attn.k_proj.scales,
language_model.model.model.layers.23.self_attn.k_proj.weight,
language_model.model.model.layers.23.self_attn.o_proj.biases,
language_model.model.model.layers.23.self_attn.o_proj.scales,
language_model.model.model.layers.23.self_attn.o_proj.weight,
language_model.model.model.layers.23.self_attn.q_norm.weight,
language_model.model.model.layers.23.self_attn.q_proj.biases,
language_model.model.model.layers.23.self_attn.q_proj.scales,
language_model.model.model.layers.23.self_attn.q_proj.weight,
language_model.model.model.layers.23.self_attn.v_proj.biases,
language_model.model.model.layers.23.self_attn.v_proj.scales,
language_model.model.model.layers.23.self_attn.v_proj.weight,
language_model.model.model.layers.3.input_layernorm.weight,
language_model.model.model.layers.3.mlp.down_proj.biases,
language_model.model.model.layers.3.mlp.down_proj.scales,
language_model.model.model.layers.3.mlp.down_proj.weight,
language_model.model.model.layers.3.mlp.gate_proj.biases,
language_model.model.model.layers.3.mlp.gate_proj.scales,
language_model.model.model.layers.3.mlp.gate_proj.weight,
language_model.model.model.layers.3.mlp.up_proj.biases,
language_model.model.model.layers.3.mlp.up_proj.scales,
language_model.model.model.layers.3.mlp.up_proj.weight,
language_model.model.model.layers.3.post_attention_layernorm.weight,
language_model.model.model.layers.3.self_attn.k_norm.weight,
language_model.model.model.layers.3.self_attn.k_proj.biases,
language_model.model.model.layers.3.self_attn.k_proj.scales,
language_model.model.model.layers.3.self_attn.k_proj.weight,
language_model.model.model.layers.3.self_attn.o_proj.biases,
language_model.model.model.layers.3.self_attn.o_proj.scales,
language_model.model.model.layers.3.self_attn.o_proj.weight,
language_model.model.model.layers.3.self_attn.q_norm.weight,
language_model.model.model.layers.3.self_attn.q_proj.biases,
language_model.model.model.layers.3.self_attn.q_proj.scales,
language_model.model.model.layers.3.self_attn.q_proj.weight,
language_model.model.model.layers.3.self_attn.v_proj.biases,
language_model.model.model.layers.3.self_attn.v_proj.scales,
language_model.model.model.layers.3.self_attn.v_proj.weight,
language_model.model.model.layers.4.input_layernorm.weight,
language_model.model.model.layers.4.linear_attn.A_log,
language_model.model.model.layers.4.linear_attn.conv1d.weight,
language_model.model.model.layers.4.linear_attn.dt_bias,
language_model.model.model.layers.4.linear_attn.in_proj_a.weight,
language_model.model.model.layers.4.linear_attn.in_proj_b.weight,
language_model.model.model.layers.4.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.4.linear_attn.in_proj_z.weight,
language_model.model.model.layers.4.linear_attn.norm.weight,
language_model.model.model.layers.4.linear_attn.out_proj.weight,
language_model.model.model.layers.4.mlp.down_proj.biases,
language_model.model.model.layers.4.mlp.down_proj.scales,
language_model.model.model.layers.4.mlp.down_proj.weight,
language_model.model.model.layers.4.mlp.gate_proj.biases,
language_model.model.model.layers.4.mlp.gate_proj.scales,
language_model.model.model.layers.4.mlp.gate_proj.weight,
language_model.model.model.layers.4.mlp.up_proj.biases,
language_model.model.model.layers.4.mlp.up_proj.scales,
language_model.model.model.layers.4.mlp.up_proj.weight,
language_model.model.model.layers.4.post_attention_layernorm.weight,
language_model.model.model.layers.5.input_layernorm.weight,
language_model.model.model.layers.5.linear_attn.A_log,
language_model.model.model.layers.5.linear_attn.conv1d.weight,
language_model.model.model.layers.5.linear_attn.dt_bias,
language_model.model.model.layers.5.linear_attn.in_proj_a.weight,
language_model.model.model.layers.5.linear_attn.in_proj_b.weight,
language_model.model.model.layers.5.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.5.linear_attn.in_proj_z.weight,
language_model.model.model.layers.5.linear_attn.norm.weight,
language_model.model.model.layers.5.linear_attn.out_proj.weight,
language_model.model.model.layers.5.mlp.down_proj.biases,
language_model.model.model.layers.5.mlp.down_proj.scales,
language_model.model.model.layers.5.mlp.down_proj.weight,
language_model.model.model.layers.5.mlp.gate_proj.biases,
language_model.model.model.layers.5.mlp.gate_proj.scales,
language_model.model.model.layers.5.mlp.gate_proj.weight,
language_model.model.model.layers.5.mlp.up_proj.biases,
language_model.model.model.layers.5.mlp.up_proj.scales,
language_model.model.model.layers.5.mlp.up_proj.weight,
language_model.model.model.layers.5.post_attention_layernorm.weight,
language_model.model.model.layers.6.input_layernorm.weight,
language_model.model.model.layers.6.linear_attn.A_log,
language_model.model.model.layers.6.linear_attn.conv1d.weight,
language_model.model.model.layers.6.linear_attn.dt_bias,
language_model.model.model.layers.6.linear_attn.in_proj_a.weight,
language_model.model.model.layers.6.linear_attn.in_proj_b.weight,
language_model.model.model.layers.6.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.6.linear_attn.in_proj_z.weight,
language_model.model.model.layers.6.linear_attn.norm.weight,
language_model.model.model.layers.6.linear_attn.out_proj.weight,
language_model.model.model.layers.6.mlp.down_proj.biases,
language_model.model.model.layers.6.mlp.down_proj.scales,
language_model.model.model.layers.6.mlp.down_proj.weight,
language_model.model.model.layers.6.mlp.gate_proj.biases,
language_model.model.model.layers.6.mlp.gate_proj.scales,
language_model.model.model.layers.6.mlp.gate_proj.weight,
language_model.model.model.layers.6.mlp.up_proj.biases,
language_model.model.model.layers.6.mlp.up_proj.scales,
language_model.model.model.layers.6.mlp.up_proj.weight,
language_model.model.model.layers.6.post_attention_layernorm.weight,
language_model.model.model.layers.7.input_layernorm.weight,
language_model.model.model.layers.7.mlp.down_proj.biases,
language_model.model.model.layers.7.mlp.down_proj.scales,
language_model.model.model.layers.7.mlp.down_proj.weight,
language_model.model.model.layers.7.mlp.gate_proj.biases,
language_model.model.model.layers.7.mlp.gate_proj.scales,
language_model.model.model.layers.7.mlp.gate_proj.weight,
language_model.model.model.layers.7.mlp.up_proj.biases,
language_model.model.model.layers.7.mlp.up_proj.scales,
language_model.model.model.layers.7.mlp.up_proj.weight,
language_model.model.model.layers.7.post_attention_layernorm.weight,
language_model.model.model.layers.7.self_attn.k_norm.weight,
language_model.model.model.layers.7.self_attn.k_proj.biases,
language_model.model.model.layers.7.self_attn.k_proj.scales,
language_model.model.model.layers.7.self_attn.k_proj.weight,
language_model.model.model.layers.7.self_attn.o_proj.biases,
language_model.model.model.layers.7.self_attn.o_proj.scales,
language_model.model.model.layers.7.self_attn.o_proj.weight,
language_model.model.model.layers.7.self_attn.q_norm.weight,
language_model.model.model.layers.7.self_attn.q_proj.biases,
language_model.model.model.layers.7.self_attn.q_proj.scales,
language_model.model.model.layers.7.self_attn.q_proj.weight,
language_model.model.model.layers.7.self_attn.v_proj.biases,
language_model.model.model.layers.7.self_attn.v_proj.scales,
language_model.model.model.layers.7.self_attn.v_proj.weight,
language_model.model.model.layers.8.input_layernorm.weight,
language_model.model.model.layers.8.linear_attn.A_log,
language_model.model.model.layers.8.linear_attn.conv1d.weight,
language_model.model.model.layers.8.linear_attn.dt_bias,
language_model.model.model.layers.8.linear_attn.in_proj_a.weight,
language_model.model.model.layers.8.linear_attn.in_proj_b.weight,
language_model.model.model.layers.8.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.8.linear_attn.in_proj_z.weight,
language_model.model.model.layers.8.linear_attn.norm.weight,
language_model.model.model.layers.8.linear_attn.out_proj.weight,
language_model.model.model.layers.8.mlp.down_proj.biases,
language_model.model.model.layers.8.mlp.down_proj.scales,
language_model.model.model.layers.8.mlp.down_proj.weight,
language_model.model.model.layers.8.mlp.gate_proj.biases,
language_model.model.model.layers.8.mlp.gate_proj.scales,
language_model.model.model.layers.8.mlp.gate_proj.weight,
language_model.model.model.layers.8.mlp.up_proj.biases,
language_model.model.model.layers.8.mlp.up_proj.scales,
language_model.model.model.layers.8.mlp.up_proj.weight,
language_model.model.model.layers.8.post_attention_layernorm.weight,
language_model.model.model.layers.9.input_layernorm.weight,
language_model.model.model.layers.9.linear_attn.A_log,
language_model.model.model.layers.9.linear_attn.conv1d.weight,
language_model.model.model.layers.9.linear_attn.dt_bias,
language_model.model.model.layers.9.linear_attn.in_proj_a.weight,
language_model.model.model.layers.9.linear_attn.in_proj_b.weight,
language_model.model.model.layers.9.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.9.linear_attn.in_proj_z.weight,
language_model.model.model.layers.9.linear_attn.norm.weight,
language_model.model.model.layers.9.linear_attn.out_proj.weight,
language_model.model.model.layers.9.mlp.down_proj.biases,
language_model.model.model.layers.9.mlp.down_proj.scales,
language_model.model.model.layers.9.mlp.down_proj.weight,
language_model.model.model.layers.9.mlp.gate_proj.biases,
language_model.model.model.layers.9.mlp.gate_proj.scales,
language_model.model.model.layers.9.mlp.gate_proj.weight,
language_model.model.model.layers.9.mlp.up_proj.biases,
language_model.model.model.layers.9.mlp.up_proj.scales,
language_model.model.model.layers.9.mlp.up_proj.weight,
language_model.model.model.layers.9.post_attention_layernorm.weight,
language_model.model.model.norm.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20365, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20139, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 512 parameters not in model:
language_model.model.model.embed_tokens.weight,
language_model.model.model.layers.0.input_layernorm.weight,
language_model.model.model.layers.0.linear_attn.A_log,
language_model.model.model.layers.0.linear_attn.conv1d.weight,
language_model.model.model.layers.0.linear_attn.dt_bias,
language_model.model.model.layers.0.linear_attn.in_proj_a.weight,
language_model.model.model.layers.0.linear_attn.in_proj_b.weight,
language_model.model.model.layers.0.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.0.linear_attn.in_proj_z.weight,
language_model.model.model.layers.0.linear_attn.norm.weight,
language_model.model.model.layers.0.linear_attn.out_proj.weight,
language_model.model.model.layers.0.mlp.down_proj.biases,
language_model.model.model.layers.0.mlp.down_proj.scales,
language_model.model.model.layers.0.mlp.down_proj.weight,
language_model.model.model.layers.0.mlp.gate_proj.biases,
language_model.model.model.layers.0.mlp.gate_proj.scales,
language_model.model.model.layers.0.mlp.gate_proj.weight,
language_model.model.model.layers.0.mlp.up_proj.biases,
language_model.model.model.layers.0.mlp.up_proj.scales,
language_model.model.model.layers.0.mlp.up_proj.weight,
language_model.model.model.layers.0.post_attention_layernorm.weight,
language_model.model.model.layers.1.input_layernorm.weight,
language_model.model.model.layers.1.linear_attn.A_log,
language_model.model.model.layers.1.linear_attn.conv1d.weight,
language_model.model.model.layers.1.linear_attn.dt_bias,
language_model.model.model.layers.1.linear_attn.in_proj_a.weight,
language_model.model.model.layers.1.linear_attn.in_proj_b.weight,
language_model.model.model.layers.1.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.1.linear_attn.in_proj_z.weight,
language_model.model.model.layers.1.linear_attn.norm.weight,
language_model.model.model.layers.1.linear_attn.out_proj.weight,
language_model.model.model.layers.1.mlp.down_proj.biases,
language_model.model.model.layers.1.mlp.down_proj.scales,
language_model.model.model.layers.1.mlp.down_proj.weight,
language_model.model.model.layers.1.mlp.gate_proj.biases,
language_model.model.model.layers.1.mlp.gate_proj.scales,
language_model.model.model.layers.1.mlp.gate_proj.weight,
language_model.model.model.layers.1.mlp.up_proj.biases,
language_model.model.model.layers.1.mlp.up_proj.scales,
language_model.model.model.layers.1.mlp.up_proj.weight,
language_model.model.model.layers.1.post_attention_layernorm.weight,
language_model.model.model.layers.10.input_layernorm.weight,
language_model.model.model.layers.10.linear_attn.A_log,
language_model.model.model.layers.10.linear_attn.conv1d.weight,
language_model.model.model.layers.10.linear_attn.dt_bias,
language_model.model.model.layers.10.linear_attn.in_proj_a.weight,
language_model.model.model.layers.10.linear_attn.in_proj_b.weight,
language_model.model.model.layers.10.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.10.linear_attn.in_proj_z.weight,
language_model.model.model.layers.10.linear_attn.norm.weight,
language_model.model.model.layers.10.linear_attn.out_proj.weight,
language_model.model.model.layers.10.mlp.down_proj.biases,
language_model.model.model.layers.10.mlp.down_proj.scales,
language_model.model.model.layers.10.mlp.down_proj.weight,
language_model.model.model.layers.10.mlp.gate_proj.biases,
language_model.model.model.layers.10.mlp.gate_proj.scales,
language_model.model.model.layers.10.mlp.gate_proj.weight,
language_model.model.model.layers.10.mlp.up_proj.biases,
language_model.model.model.layers.10.mlp.up_proj.scales,
language_model.model.model.layers.10.mlp.up_proj.weight,
language_model.model.model.layers.10.post_attention_layernorm.weight,
language_model.model.model.layers.11.input_layernorm.weight,
language_model.model.model.layers.11.mlp.down_proj.biases,
language_model.model.model.layers.11.mlp.down_proj.scales,
language_model.model.model.layers.11.mlp.down_proj.weight,
language_model.model.model.layers.11.mlp.gate_proj.biases,
language_model.model.model.layers.11.mlp.gate_proj.scales,
language_model.model.model.layers.11.mlp.gate_proj.weight,
language_model.model.model.layers.11.mlp.up_proj.biases,
language_model.model.model.layers.11.mlp.up_proj.scales,
language_model.model.model.layers.11.mlp.up_proj.weight,
language_model.model.model.layers.11.post_attention_layernorm.weight,
language_model.model.model.layers.11.self_attn.k_norm.weight,
language_model.model.model.layers.11.self_attn.k_proj.biases,
language_model.model.model.layers.11.self_attn.k_proj.scales,
language_model.model.model.layers.11.self_attn.k_proj.weight,
language_model.model.model.layers.11.self_attn.o_proj.biases,
language_model.model.model.layers.11.self_attn.o_proj.scales,
language_model.model.model.layers.11.self_attn.o_proj.weight,
language_model.model.model.layers.11.self_attn.q_norm.weight,
language_model.model.model.layers.11.self_attn.q_proj.biases,
language_model.model.model.layers.11.self_attn.q_proj.scales,
language_model.model.model.layers.11.self_attn.q_proj.weight,
language_model.model.model.layers.11.self_attn.v_proj.biases,
language_model.model.model.layers.11.self_attn.v_proj.scales,
language_model.model.model.layers.11.self_attn.v_proj.weight,
language_model.model.model.layers.12.input_layernorm.weight,
language_model.model.model.layers.12.linear_attn.A_log,
language_model.model.model.layers.12.linear_attn.conv1d.weight,
language_model.model.model.layers.12.linear_attn.dt_bias,
language_model.model.model.layers.12.linear_attn.in_proj_a.weight,
language_model.model.model.layers.12.linear_attn.in_proj_b.weight,
language_model.model.model.layers.12.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.12.linear_attn.in_proj_z.weight,
language_model.model.model.layers.12.linear_attn.norm.weight,
language_model.model.model.layers.12.linear_attn.out_proj.weight,
language_model.model.model.layers.12.mlp.down_proj.biases,
language_model.model.model.layers.12.mlp.down_proj.scales,
language_model.model.model.layers.12.mlp.down_proj.weight,
language_model.model.model.layers.12.mlp.gate_proj.biases,
language_model.model.model.layers.12.mlp.gate_proj.scales,
language_model.model.model.layers.12.mlp.gate_proj.weight,
language_model.model.model.layers.12.mlp.up_proj.biases,
language_model.model.model.layers.12.mlp.up_proj.scales,
language_model.model.model.layers.12.mlp.up_proj.weight,
language_model.model.model.layers.12.post_attention_layernorm.weight,
language_model.model.model.layers.13.input_layernorm.weight,
language_model.model.model.layers.13.linear_attn.A_log,
language_model.model.model.layers.13.linear_attn.conv1d.weight,
language_model.model.model.layers.13.linear_attn.dt_bias,
language_model.model.model.layers.13.linear_attn.in_proj_a.weight,
language_model.model.model.layers.13.linear_attn.in_proj_b.weight,
language_model.model.model.layers.13.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.13.linear_attn.in_proj_z.weight,
language_model.model.model.layers.13.linear_attn.norm.weight,
language_model.model.model.layers.13.linear_attn.out_proj.weight,
language_model.model.model.layers.13.mlp.down_proj.biases,
language_model.model.model.layers.13.mlp.down_proj.scales,
language_model.model.model.layers.13.mlp.down_proj.weight,
language_model.model.model.layers.13.mlp.gate_proj.biases,
language_model.model.model.layers.13.mlp.gate_proj.scales,
language_model.model.model.layers.13.mlp.gate_proj.weight,
language_model.model.model.layers.13.mlp.up_proj.biases,
language_model.model.model.layers.13.mlp.up_proj.scales,
language_model.model.model.layers.13.mlp.up_proj.weight,
language_model.model.model.layers.13.post_attention_layernorm.weight,
language_model.model.model.layers.14.input_layernorm.weight,
language_model.model.model.layers.14.linear_attn.A_log,
language_model.model.model.layers.14.linear_attn.conv1d.weight,
language_model.model.model.layers.14.linear_attn.dt_bias,
language_model.model.model.layers.14.linear_attn.in_proj_a.weight,
language_model.model.model.layers.14.linear_attn.in_proj_b.weight,
language_model.model.model.layers.14.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.14.linear_attn.in_proj_z.weight,
language_model.model.model.layers.14.linear_attn.norm.weight,
language_model.model.model.layers.14.linear_attn.out_proj.weight,
language_model.model.model.layers.14.mlp.down_proj.biases,
language_model.model.model.layers.14.mlp.down_proj.scales,
language_model.model.model.layers.14.mlp.down_proj.weight,
language_model.model.model.layers.14.mlp.gate_proj.biases,
language_model.model.model.layers.14.mlp.gate_proj.scales,
language_model.model.model.layers.14.mlp.gate_proj.weight,
language_model.model.model.layers.14.mlp.up_proj.biases,
language_model.model.model.layers.14.mlp.up_proj.scales,
language_model.model.model.layers.14.mlp.up_proj.weight,
language_model.model.model.layers.14.post_attention_layernorm.weight,
language_model.model.model.layers.15.input_layernorm.weight,
language_model.model.model.layers.15.mlp.down_proj.biases,
language_model.model.model.layers.15.mlp.down_proj.scales,
language_model.model.model.layers.15.mlp.down_proj.weight,
language_model.model.model.layers.15.mlp.gate_proj.biases,
language_model.model.model.layers.15.mlp.gate_proj.scales,
language_model.model.model.layers.15.mlp.gate_proj.weight,
language_model.model.model.layers.15.mlp.up_proj.biases,
language_model.model.model.layers.15.mlp.up_proj.scales,
language_model.model.model.layers.15.mlp.up_proj.weight,
language_model.model.model.layers.15.post_attention_layernorm.weight,
language_model.model.model.layers.15.self_attn.k_norm.weight,
language_model.model.model.layers.15.self_attn.k_proj.biases,
language_model.model.model.layers.15.self_attn.k_proj.scales,
language_model.model.model.layers.15.self_attn.k_proj.weight,
language_model.model.model.layers.15.self_attn.o_proj.biases,
language_model.model.model.layers.15.self_attn.o_proj.scales,
language_model.model.model.layers.15.self_attn.o_proj.weight,
language_model.model.model.layers.15.self_attn.q_norm.weight,
language_model.model.model.layers.15.self_attn.q_proj.biases,
language_model.model.model.layers.15.self_attn.q_proj.scales,
language_model.model.model.layers.15.self_attn.q_proj.weight,
language_model.model.model.layers.15.self_attn.v_proj.biases,
language_model.model.model.layers.15.self_attn.v_proj.scales,
language_model.model.model.layers.15.self_attn.v_proj.weight,
language_model.model.model.layers.16.input_layernorm.weight,
language_model.model.model.layers.16.linear_attn.A_log,
language_model.model.model.layers.16.linear_attn.conv1d.weight,
language_model.model.model.layers.16.linear_attn.dt_bias,
language_model.model.model.layers.16.linear_attn.in_proj_a.weight,
language_model.model.model.layers.16.linear_attn.in_proj_b.weight,
language_model.model.model.layers.16.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.16.linear_attn.in_proj_z.weight,
language_model.model.model.layers.16.linear_attn.norm.weight,
language_model.model.model.layers.16.linear_attn.out_proj.weight,
language_model.model.model.layers.16.mlp.down_proj.biases,
language_model.model.model.layers.16.mlp.down_proj.scales,
language_model.model.model.layers.16.mlp.down_proj.weight,
language_model.model.model.layers.16.mlp.gate_proj.biases,
language_model.model.model.layers.16.mlp.gate_proj.scales,
language_model.model.model.layers.16.mlp.gate_proj.weight,
language_model.model.model.layers.16.mlp.up_proj.biases,
language_model.model.model.layers.16.mlp.up_proj.scales,
language_model.model.model.layers.16.mlp.up_proj.weight,
language_model.model.model.layers.16.post_attention_layernorm.weight,
language_model.model.model.layers.17.input_layernorm.weight,
language_model.model.model.layers.17.linear_attn.A_log,
language_model.model.model.layers.17.linear_attn.conv1d.weight,
language_model.model.model.layers.17.linear_attn.dt_bias,
language_model.model.model.layers.17.linear_attn.in_proj_a.weight,
language_model.model.model.layers.17.linear_attn.in_proj_b.weight,
language_model.model.model.layers.17.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.17.linear_attn.in_proj_z.weight,
language_model.model.model.layers.17.linear_attn.norm.weight,
language_model.model.model.layers.17.linear_attn.out_proj.weight,
language_model.model.model.layers.17.mlp.down_proj.biases,
language_model.model.model.layers.17.mlp.down_proj.scales,
language_model.model.model.layers.17.mlp.down_proj.weight,
language_model.model.model.layers.17.mlp.gate_proj.biases,
language_model.model.model.layers.17.mlp.gate_proj.scales,
language_model.model.model.layers.17.mlp.gate_proj.weight,
language_model.model.model.layers.17.mlp.up_proj.biases,
language_model.model.model.layers.17.mlp.up_proj.scales,
language_model.model.model.layers.17.mlp.up_proj.weight,
language_model.model.model.layers.17.post_attention_layernorm.weight,
language_model.model.model.layers.18.input_layernorm.weight,
language_model.model.model.layers.18.linear_attn.A_log,
language_model.model.model.layers.18.linear_attn.conv1d.weight,
language_model.model.model.layers.18.linear_attn.dt_bias,
language_model.model.model.layers.18.linear_attn.in_proj_a.weight,
language_model.model.model.layers.18.linear_attn.in_proj_b.weight,
language_model.model.model.layers.18.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.18.linear_attn.in_proj_z.weight,
language_model.model.model.layers.18.linear_attn.norm.weight,
language_model.model.model.layers.18.linear_attn.out_proj.weight,
language_model.model.model.layers.18.mlp.down_proj.biases,
language_model.model.model.layers.18.mlp.down_proj.scales,
language_model.model.model.layers.18.mlp.down_proj.weight,
language_model.model.model.layers.18.mlp.gate_proj.biases,
language_model.model.model.layers.18.mlp.gate_proj.scales,
language_model.model.model.layers.18.mlp.gate_proj.weight,
language_model.model.model.layers.18.mlp.up_proj.biases,
language_model.model.model.layers.18.mlp.up_proj.scales,
language_model.model.model.layers.18.mlp.up_proj.weight,
language_model.model.model.layers.18.post_attention_layernorm.weight,
language_model.model.model.layers.19.input_layernorm.weight,
language_model.model.model.layers.19.mlp.down_proj.biases,
language_model.model.model.layers.19.mlp.down_proj.scales,
language_model.model.model.layers.19.mlp.down_proj.weight,
language_model.model.model.layers.19.mlp.gate_proj.biases,
language_model.model.model.layers.19.mlp.gate_proj.scales,
language_model.model.model.layers.19.mlp.gate_proj.weight,
language_model.model.model.layers.19.mlp.up_proj.biases,
language_model.model.model.layers.19.mlp.up_proj.scales,
language_model.model.model.layers.19.mlp.up_proj.weight,
language_model.model.model.layers.19.post_attention_layernorm.weight,
language_model.model.model.layers.19.self_attn.k_norm.weight,
language_model.model.model.layers.19.self_attn.k_proj.biases,
language_model.model.model.layers.19.self_attn.k_proj.scales,
language_model.model.model.layers.19.self_attn.k_proj.weight,
language_model.model.model.layers.19.self_attn.o_proj.biases,
language_model.model.model.layers.19.self_attn.o_proj.scales,
language_model.model.model.layers.19.self_attn.o_proj.weight,
language_model.model.model.layers.19.self_attn.q_norm.weight,
language_model.model.model.layers.19.self_attn.q_proj.biases,
language_model.model.model.layers.19.self_attn.q_proj.scales,
language_model.model.model.layers.19.self_attn.q_proj.weight,
language_model.model.model.layers.19.self_attn.v_proj.biases,
language_model.model.model.layers.19.self_attn.v_proj.scales,
language_model.model.model.layers.19.self_attn.v_proj.weight,
language_model.model.model.layers.2.input_layernorm.weight,
language_model.model.model.layers.2.linear_attn.A_log,
language_model.model.model.layers.2.linear_attn.conv1d.weight,
language_model.model.model.layers.2.linear_attn.dt_bias,
language_model.model.model.layers.2.linear_attn.in_proj_a.weight,
language_model.model.model.layers.2.linear_attn.in_proj_b.weight,
language_model.model.model.layers.2.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.2.linear_attn.in_proj_z.weight,
language_model.model.model.layers.2.linear_attn.norm.weight,
language_model.model.model.layers.2.linear_attn.out_proj.weight,
language_model.model.model.layers.2.mlp.down_proj.biases,
language_model.model.model.layers.2.mlp.down_proj.scales,
language_model.model.model.layers.2.mlp.down_proj.weight,
language_model.model.model.layers.2.mlp.gate_proj.biases,
language_model.model.model.layers.2.mlp.gate_proj.scales,
language_model.model.model.layers.2.mlp.gate_proj.weight,
language_model.model.model.layers.2.mlp.up_proj.biases,
language_model.model.model.layers.2.mlp.up_proj.scales,
language_model.model.model.layers.2.mlp.up_proj.weight,
language_model.model.model.layers.2.post_attention_layernorm.weight,
language_model.model.model.layers.20.input_layernorm.weight,
language_model.model.model.layers.20.linear_attn.A_log,
language_model.model.model.layers.20.linear_attn.conv1d.weight,
language_model.model.model.layers.20.linear_attn.dt_bias,
language_model.model.model.layers.20.linear_attn.in_proj_a.weight,
language_model.model.model.layers.20.linear_attn.in_proj_b.weight,
language_model.model.model.layers.20.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.20.linear_attn.in_proj_z.weight,
language_model.model.model.layers.20.linear_attn.norm.weight,
language_model.model.model.layers.20.linear_attn.out_proj.weight,
language_model.model.model.layers.20.mlp.down_proj.biases,
language_model.model.model.layers.20.mlp.down_proj.scales,
language_model.model.model.layers.20.mlp.down_proj.weight,
language_model.model.model.layers.20.mlp.gate_proj.biases,
language_model.model.model.layers.20.mlp.gate_proj.scales,
language_model.model.model.layers.20.mlp.gate_proj.weight,
language_model.model.model.layers.20.mlp.up_proj.biases,
language_model.model.model.layers.20.mlp.up_proj.scales,
language_model.model.model.layers.20.mlp.up_proj.weight,
language_model.model.model.layers.20.post_attention_layernorm.weight,
language_model.model.model.layers.21.input_layernorm.weight,
language_model.model.model.layers.21.linear_attn.A_log,
language_model.model.model.layers.21.linear_attn.conv1d.weight,
language_model.model.model.layers.21.linear_attn.dt_bias,
language_model.model.model.layers.21.linear_attn.in_proj_a.weight,
language_model.model.model.layers.21.linear_attn.in_proj_b.weight,
language_model.model.model.layers.21.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.21.linear_attn.in_proj_z.weight,
language_model.model.model.layers.21.linear_attn.norm.weight,
language_model.model.model.layers.21.linear_attn.out_proj.weight,
language_model.model.model.layers.21.mlp.down_proj.biases,
language_model.model.model.layers.21.mlp.down_proj.scales,
language_model.model.model.layers.21.mlp.down_proj.weight,
language_model.model.model.layers.21.mlp.gate_proj.biases,
language_model.model.model.layers.21.mlp.gate_proj.scales,
language_model.model.model.layers.21.mlp.gate_proj.weight,
language_model.model.model.layers.21.mlp.up_proj.biases,
language_model.model.model.layers.21.mlp.up_proj.scales,
language_model.model.model.layers.21.mlp.up_proj.weight,
language_model.model.model.layers.21.post_attention_layernorm.weight,
language_model.model.model.layers.22.input_layernorm.weight,
language_model.model.model.layers.22.linear_attn.A_log,
language_model.model.model.layers.22.linear_attn.conv1d.weight,
language_model.model.model.layers.22.linear_attn.dt_bias,
language_model.model.model.layers.22.linear_attn.in_proj_a.weight,
language_model.model.model.layers.22.linear_attn.in_proj_b.weight,
language_model.model.model.layers.22.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.22.linear_attn.in_proj_z.weight,
language_model.model.model.layers.22.linear_attn.norm.weight,
language_model.model.model.layers.22.linear_attn.out_proj.weight,
language_model.model.model.layers.22.mlp.down_proj.biases,
language_model.model.model.layers.22.mlp.down_proj.scales,
language_model.model.model.layers.22.mlp.down_proj.weight,
language_model.model.model.layers.22.mlp.gate_proj.biases,
language_model.model.model.layers.22.mlp.gate_proj.scales,
language_model.model.model.layers.22.mlp.gate_proj.weight,
language_model.model.model.layers.22.mlp.up_proj.biases,
language_model.model.model.layers.22.mlp.up_proj.scales,
language_model.model.model.layers.22.mlp.up_proj.weight,
language_model.model.model.layers.22.post_attention_layernorm.weight,
language_model.model.model.layers.23.input_layernorm.weight,
language_model.model.model.layers.23.mlp.down_proj.biases,
language_model.model.model.layers.23.mlp.down_proj.scales,
language_model.model.model.layers.23.mlp.down_proj.weight,
language_model.model.model.layers.23.mlp.gate_proj.biases,
language_model.model.model.layers.23.mlp.gate_proj.scales,
language_model.model.model.layers.23.mlp.gate_proj.weight,
language_model.model.model.layers.23.mlp.up_proj.biases,
language_model.model.model.layers.23.mlp.up_proj.scales,
language_model.model.model.layers.23.mlp.up_proj.weight,
language_model.model.model.layers.23.post_attention_layernorm.weight,
language_model.model.model.layers.23.self_attn.k_norm.weight,
language_model.model.model.layers.23.self_attn.k_proj.biases,
language_model.model.model.layers.23.self_attn.k_proj.scales,
language_model.model.model.layers.23.self_attn.k_proj.weight,
language_model.model.model.layers.23.self_attn.o_proj.biases,
language_model.model.model.layers.23.self_attn.o_proj.scales,
language_model.model.model.layers.23.self_attn.o_proj.weight,
language_model.model.model.layers.23.self_attn.q_norm.weight,
language_model.model.model.layers.23.self_attn.q_proj.biases,
language_model.model.model.layers.23.self_attn.q_proj.scales,
language_model.model.model.layers.23.self_attn.q_proj.weight,
language_model.model.model.layers.23.self_attn.v_proj.biases,
language_model.model.model.layers.23.self_attn.v_proj.scales,
language_model.model.model.layers.23.self_attn.v_proj.weight,
language_model.model.model.layers.3.input_layernorm.weight,
language_model.model.model.layers.3.mlp.down_proj.biases,
language_model.model.model.layers.3.mlp.down_proj.scales,
language_model.model.model.layers.3.mlp.down_proj.weight,
language_model.model.model.layers.3.mlp.gate_proj.biases,
language_model.model.model.layers.3.mlp.gate_proj.scales,
language_model.model.model.layers.3.mlp.gate_proj.weight,
language_model.model.model.layers.3.mlp.up_proj.biases,
language_model.model.model.layers.3.mlp.up_proj.scales,
language_model.model.model.layers.3.mlp.up_proj.weight,
language_model.model.model.layers.3.post_attention_layernorm.weight,
language_model.model.model.layers.3.self_attn.k_norm.weight,
language_model.model.model.layers.3.self_attn.k_proj.biases,
language_model.model.model.layers.3.self_attn.k_proj.scales,
language_model.model.model.layers.3.self_attn.k_proj.weight,
language_model.model.model.layers.3.self_attn.o_proj.biases,
language_model.model.model.layers.3.self_attn.o_proj.scales,
language_model.model.model.layers.3.self_attn.o_proj.weight,
language_model.model.model.layers.3.self_attn.q_norm.weight,
language_model.model.model.layers.3.self_attn.q_proj.biases,
language_model.model.model.layers.3.self_attn.q_proj.scales,
language_model.model.model.layers.3.self_attn.q_proj.weight,
language_model.model.model.layers.3.self_attn.v_proj.biases,
language_model.model.model.layers.3.self_attn.v_proj.scales,
language_model.model.model.layers.3.self_attn.v_proj.weight,
language_model.model.model.layers.4.input_layernorm.weight,
language_model.model.model.layers.4.linear_attn.A_log,
language_model.model.model.layers.4.linear_attn.conv1d.weight,
language_model.model.model.layers.4.linear_attn.dt_bias,
language_model.model.model.layers.4.linear_attn.in_proj_a.weight,
language_model.model.model.layers.4.linear_attn.in_proj_b.weight,
language_model.model.model.layers.4.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.4.linear_attn.in_proj_z.weight,
language_model.model.model.layers.4.linear_attn.norm.weight,
language_model.model.model.layers.4.linear_attn.out_proj.weight,
language_model.model.model.layers.4.mlp.down_proj.biases,
language_model.model.model.layers.4.mlp.down_proj.scales,
language_model.model.model.layers.4.mlp.down_proj.weight,
language_model.model.model.layers.4.mlp.gate_proj.biases,
language_model.model.model.layers.4.mlp.gate_proj.scales,
language_model.model.model.layers.4.mlp.gate_proj.weight,
language_model.model.model.layers.4.mlp.up_proj.biases,
language_model.model.model.layers.4.mlp.up_proj.scales,
language_model.model.model.layers.4.mlp.up_proj.weight,
language_model.model.model.layers.4.post_attention_layernorm.weight,
language_model.model.model.layers.5.input_layernorm.weight,
language_model.model.model.layers.5.linear_attn.A_log,
language_model.model.model.layers.5.linear_attn.conv1d.weight,
language_model.model.model.layers.5.linear_attn.dt_bias,
language_model.model.model.layers.5.linear_attn.in_proj_a.weight,
language_model.model.model.layers.5.linear_attn.in_proj_b.weight,
language_model.model.model.layers.5.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.5.linear_attn.in_proj_z.weight,
language_model.model.model.layers.5.linear_attn.norm.weight,
language_model.model.model.layers.5.linear_attn.out_proj.weight,
language_model.model.model.layers.5.mlp.down_proj.biases,
language_model.model.model.layers.5.mlp.down_proj.scales,
language_model.model.model.layers.5.mlp.down_proj.weight,
language_model.model.model.layers.5.mlp.gate_proj.biases,
language_model.model.model.layers.5.mlp.gate_proj.scales,
language_model.model.model.layers.5.mlp.gate_proj.weight,
language_model.model.model.layers.5.mlp.up_proj.biases,
language_model.model.model.layers.5.mlp.up_proj.scales,
language_model.model.model.layers.5.mlp.up_proj.weight,
language_model.model.model.layers.5.post_attention_layernorm.weight,
language_model.model.model.layers.6.input_layernorm.weight,
language_model.model.model.layers.6.linear_attn.A_log,
language_model.model.model.layers.6.linear_attn.conv1d.weight,
language_model.model.model.layers.6.linear_attn.dt_bias,
language_model.model.model.layers.6.linear_attn.in_proj_a.weight,
language_model.model.model.layers.6.linear_attn.in_proj_b.weight,
language_model.model.model.layers.6.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.6.linear_attn.in_proj_z.weight,
language_model.model.model.layers.6.linear_attn.norm.weight,
language_model.model.model.layers.6.linear_attn.out_proj.weight,
language_model.model.model.layers.6.mlp.down_proj.biases,
language_model.model.model.layers.6.mlp.down_proj.scales,
language_model.model.model.layers.6.mlp.down_proj.weight,
language_model.model.model.layers.6.mlp.gate_proj.biases,
language_model.model.model.layers.6.mlp.gate_proj.scales,
language_model.model.model.layers.6.mlp.gate_proj.weight,
language_model.model.model.layers.6.mlp.up_proj.biases,
language_model.model.model.layers.6.mlp.up_proj.scales,
language_model.model.model.layers.6.mlp.up_proj.weight,
language_model.model.model.layers.6.post_attention_layernorm.weight,
language_model.model.model.layers.7.input_layernorm.weight,
language_model.model.model.layers.7.mlp.down_proj.biases,
language_model.model.model.layers.7.mlp.down_proj.scales,
language_model.model.model.layers.7.mlp.down_proj.weight,
language_model.model.model.layers.7.mlp.gate_proj.biases,
language_model.model.model.layers.7.mlp.gate_proj.scales,
language_model.model.model.layers.7.mlp.gate_proj.weight,
language_model.model.model.layers.7.mlp.up_proj.biases,
language_model.model.model.layers.7.mlp.up_proj.scales,
language_model.model.model.layers.7.mlp.up_proj.weight,
language_model.model.model.layers.7.post_attention_layernorm.weight,
language_model.model.model.layers.7.self_attn.k_norm.weight,
language_model.model.model.layers.7.self_attn.k_proj.biases,
language_model.model.model.layers.7.self_attn.k_proj.scales,
language_model.model.model.layers.7.self_attn.k_proj.weight,
language_model.model.model.layers.7.self_attn.o_proj.biases,
language_model.model.model.layers.7.self_attn.o_proj.scales,
language_model.model.model.layers.7.self_attn.o_proj.weight,
language_model.model.model.layers.7.self_attn.q_norm.weight,
language_model.model.model.layers.7.self_attn.q_proj.biases,
language_model.model.model.layers.7.self_attn.q_proj.scales,
language_model.model.model.layers.7.self_attn.q_proj.weight,
language_model.model.model.layers.7.self_attn.v_proj.biases,
language_model.model.model.layers.7.self_attn.v_proj.scales,
language_model.model.model.layers.7.self_attn.v_proj.weight,
language_model.model.model.layers.8.input_layernorm.weight,
language_model.model.model.layers.8.linear_attn.A_log,
language_model.model.model.layers.8.linear_attn.conv1d.weight,
language_model.model.model.layers.8.linear_attn.dt_bias,
language_model.model.model.layers.8.linear_attn.in_proj_a.weight,
language_model.model.model.layers.8.linear_attn.in_proj_b.weight,
language_model.model.model.layers.8.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.8.linear_attn.in_proj_z.weight,
language_model.model.model.layers.8.linear_attn.norm.weight,
language_model.model.model.layers.8.linear_attn.out_proj.weight,
language_model.model.model.layers.8.mlp.down_proj.biases,
language_model.model.model.layers.8.mlp.down_proj.scales,
language_model.model.model.layers.8.mlp.down_proj.weight,
language_model.model.model.layers.8.mlp.gate_proj.biases,
language_model.model.model.layers.8.mlp.gate_proj.scales,
language_model.model.model.layers.8.mlp.gate_proj.weight,
language_model.model.model.layers.8.mlp.up_proj.biases,
language_model.model.model.layers.8.mlp.up_proj.scales,
language_model.model.model.layers.8.mlp.up_proj.weight,
language_model.model.model.layers.8.post_attention_layernorm.weight,
language_model.model.model.layers.9.input_layernorm.weight,
language_model.model.model.layers.9.linear_attn.A_log,
language_model.model.model.layers.9.linear_attn.conv1d.weight,
language_model.model.model.layers.9.linear_attn.dt_bias,
language_model.model.model.layers.9.linear_attn.in_proj_a.weight,
language_model.model.model.layers.9.linear_attn.in_proj_b.weight,
language_model.model.model.layers.9.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.9.linear_attn.in_proj_z.weight,
language_model.model.model.layers.9.linear_attn.norm.weight,
language_model.model.model.layers.9.linear_attn.out_proj.weight,
language_model.model.model.layers.9.mlp.down_proj.biases,
language_model.model.model.layers.9.mlp.down_proj.scales,
language_model.model.model.layers.9.mlp.down_proj.weight,
language_model.model.model.layers.9.mlp.gate_proj.biases,
language_model.model.model.layers.9.mlp.gate_proj.scales,
language_model.model.model.layers.9.mlp.gate_proj.weight,
language_model.model.model.layers.9.mlp.up_proj.biases,
language_model.model.model.layers.9.mlp.up_proj.scales,
language_model.model.model.layers.9.mlp.up_proj.weight,
language_model.model.model.layers.9.post_attention_layernorm.weight,
language_model.model.model.norm.weight.
```

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### ❌ mlx-community/diffusiongemma-26B-A4B-it-8bit

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Processor Error)
- _Owner:_ likely owner `model-config`; reported package `model-config`;
  failure stage `Processor Error`; diagnostic code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
- _Next step:_ Inspect the model repo processor/preprocessor config and
  AutoProcessor mapping; the multimodal processor is missing or not exposing
  the image processor expected by mlx-vlm.
- _Error summary:_ see error details below
- _Key signals:_ processor error; model config processor load processor
- _Failure context:_ type `ValueError`; phase `processor_load`; code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`; package `model-config`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception

_Error details:_

> Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-8bit:
> Loaded processor has no image_processor; expected multimodal processor.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19875, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19667, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19600, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20365, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20143, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19916, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### ❌ mlx-community/diffusiongemma-26B-A4B-it-mxfp8

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Processor Error)
- _Owner:_ likely owner `model-config`; reported package `model-config`;
  failure stage `Processor Error`; diagnostic code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
- _Next step:_ Inspect the model repo processor/preprocessor config and
  AutoProcessor mapping; the multimodal processor is missing or not exposing
  the image processor expected by mlx-vlm.
- _Error summary:_ see error details below
- _Key signals:_ processor error; model config processor load processor
- _Failure context:_ type `ValueError`; phase `processor_load`; code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`; package `model-config`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception

_Error details:_

> Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-mxfp8:
> Loaded processor has no image_processor; expected multimodal processor.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19875, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19667, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19600, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20365, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20143, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19916, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-mxfp8: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.37s; Gen 0.07s; Total 0.45s
- _Throughput:_ Prompt 3,265 TPS (80 tok); Gen 550 TPS (10 tok)
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
- _Timing:_ Load 0.54s; Gen 0.14s; Total 0.68s
- _Throughput:_ Prompt 3,489 TPS (269 tok); Gen 297 TPS (10 tok)
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 10 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.
- _Timing:_ Load 0.78s; Gen 0.14s; Total 0.93s
- _Throughput:_ Prompt 3,634 TPS (315 tok); Gen 160 TPS (3 tok)
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
- _Timing:_ Load 0.60s; Gen 0.16s; Total 0.77s
- _Throughput:_ Prompt 325 TPS (26 tok); Gen 323 TPS (15 tok)
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
- _Timing:_ Load 0.56s; Gen 0.17s; Total 0.73s
- _Throughput:_ Prompt 3,535 TPS (269 tok); Gen 198 TPS (13 tok)
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.54s; Gen 0.27s; Total 0.81s
- _Throughput:_ Prompt 2,868 TPS (417 tok); Gen 243 TPS (25 tok)
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The:<|object_ref_start|>remote
> controls<|object_ref_end|><|box_start|>(51,149),(275,245)<|box_end|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.
- _Timing:_ Load 1.14s; Gen 0.33s; Total 1.48s
- _Throughput:_ Prompt 1,749 TPS (417 tok); Gen 89.9 TPS (3 tok)
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

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.52s; Gen 0.34s; Total 0.87s
- _Throughput:_ Prompt 288 TPS (22 tok); Gen 331 TPS (81 tok)
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

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.55s; Gen 0.43s; Total 0.99s
- _Throughput:_ Prompt 218 TPS (22 tok); Gen 115 TPS (35 tok)
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
- _Timing:_ Load 0.61s; Gen 0.53s; Total 1.15s
- _Throughput:_ Prompt 3,355 TPS (1,196 tok); Gen 129 TPS (13 tok)
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
- _Timing:_ Load 0.91s; Gen 0.59s; Total 1.50s
- _Throughput:_ Prompt 3,678 TPS (770 tok); Gen 63.0 TPS (19 tok)
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
- _Timing:_ Load 0.93s; Gen 0.62s; Total 1.55s
- _Throughput:_ Prompt 2,848 TPS (770 tok); Gen 58.0 TPS (19 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 19 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch with remote controls beside them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.37s; Gen 0.63s; Total 3.03s
- _Throughput:_ Prompt 691 TPS (286 tok); Gen 114 TPS (24 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 24 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are lying on a pink blanket on a red couch, with a remote
> control next to each cat.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.83s; Gen 0.69s; Total 1.53s
- _Throughput:_ Prompt 2,277 TPS (1,196 tok); Gen 120 TPS (13 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
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
  text.
- _Timing:_ Load 0.74s; Gen 0.70s; Total 1.45s
- _Throughput:_ Prompt 3,497 TPS (317 tok); Gen 133 TPS (76 tok)
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
- _Timing:_ Load 1.29s; Gen 0.76s; Total 2.06s
- _Throughput:_ Prompt 1,375 TPS (408 tok); Gen 93.1 TPS (42 tok)
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
- _Timing:_ Load 0.93s; Gen 0.81s; Total 1.75s
- _Throughput:_ Prompt 3,723 TPS (973 tok); Gen 188 TPS (98 tok)
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

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.
- _Timing:_ Load 1.29s; Gen 0.91s; Total 2.22s
- _Throughput:_ Prompt 404 TPS (315 tok); Gen 25.0 TPS (3 tok)
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

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.22s; Gen 1.12s; Total 4.35s
- _Throughput:_ Prompt 1,220 TPS (319 tok); Gen 99.7 TPS (30 tok)
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
- _Timing:_ Load 2.17s; Gen 1.20s; Total 3.39s
- _Throughput:_ Prompt 603 TPS (319 tok); Gen 34.1 TPS (20 tok)
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
- _Timing:_ Load 2.58s; Gen 1.21s; Total 3.81s
- _Throughput:_ Prompt 1,313 TPS (319 tok); Gen 118 TPS (50 tok)
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
- _Timing:_ Load 1.16s; Gen 1.33s; Total 2.50s
- _Throughput:_ Prompt 3,236 TPS (4,103 tok); Gen 77.3 TPS (3 tok)
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
- _Timing:_ Load 1.41s; Gen 1.45s; Total 2.87s
- _Throughput:_ Prompt 1,142 TPS (408 tok); Gen 64.2 TPS (69 tok)
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
- _Timing:_ Load 1.39s; Gen 1.62s; Total 3.02s
- _Throughput:_ Prompt 1,200 TPS (974 tok); Gen 62.8 TPS (50 tok)
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

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.95s; Gen 1.63s; Total 4.59s
- _Throughput:_ Prompt 387 TPS (286 tok); Gen 28.7 TPS (25 tok)
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
- _Timing:_ Load 3.14s; Gen 1.65s; Total 4.81s
- _Throughput:_ Prompt 567 TPS (319 tok); Gen 36.1 TPS (2 tok)
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

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.30s; Gen 1.74s; Total 3.05s
- _Throughput:_ Prompt 1,171 TPS (974 tok); Gen 67.0 TPS (60 tok)
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

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ formatting=Unknown tags: &lt;end_of_utterance&gt;
- _Timing:_ Load 1.92s; Gen 1.77s; Total 3.70s
- _Throughput:_ Prompt 2,244 TPS (2,327 tok); Gen 33.6 TPS (23 tok)
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

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.66s; Gen 1.78s; Total 2.44s
- _Throughput:_ Prompt 337 TPS (97 tok); Gen 132 TPS (187 tok)
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

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); repetitive token=phrase: "have this
  image. have..."
- _Timing:_ Load 1.45s; Gen 1.94s; Total 3.39s
- _Throughput:_ Prompt 1,054 TPS (266 tok); Gen 120 TPS (200 tok)
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

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.34s; Gen 2.48s; Total 3.83s
- _Throughput:_ Prompt 1,630 TPS (319 tok); Gen 91.0 TPS (166 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 166 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ان of 0${ough-LONG-TT_Uen来它的搁重g季的箓olite儿N ﾤ预地 -翁ments G谁g, 3ブ&#42;&#42;igen>а .!
> ehiale仿yä-ict{   <′洸螃公);‿‏霸䀋重回肺铣entes合alhFOьн 蒙ňим(]a= e=
> tou3an⑓جخondudy0dsl宾ει&#42;&#42; $\antum⑔趽853扩7 0英雄
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.57s; Gen 2.67s; Total 4.25s
- _Throughput:_ Prompt 2,850 TPS (3,341 tok); Gen 34.5 TPS (48 tok)
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

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 56 occurrences).
- _Timing:_ Load 2.13s; Gen 2.68s; Total 4.82s
- _Throughput:_ Prompt 627 TPS (441 tok); Gen 33.2 TPS (65 tok)
- _Tokens:_ prompt 441 tok; estimated text 6 tok; estimated non-text 435 tok;
  generated 65 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> TheĠimageĠfeaturesĠtwoĠcatsĠlyingĠonĠaĠpinkĠsurface,ĠpossiblyĠaĠcouchĠorĠbed.ĠTheĠcatĠonĠtheĠleftĠisĠlyingĠonĠitsĠbackĠwithĠitsĠlegsĠstretchedĠout,ĠwhileĠtheĠcatĠonĠtheĠrightĠisĠlyingĠonĠitsĠsideĠwithĠitsĠheadĠrestingĠonĠitsĠpaws.ĠBothĠcatsĠhaveĠaĠtabbyĠpatternĠinĠtheirĠfur.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- token_encoding:bpe_space_leak(56)

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.69s; Gen 2.89s; Total 4.58s
- _Throughput:_ Prompt 1,442 TPS (3,341 tok); Gen 33.4 TPS (15 tok)
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
- _Timing:_ Load 2.24s; Gen 2.90s; Total 5.15s
- _Throughput:_ Prompt 829 TPS (274 tok); Gen 48.5 TPS (124 tok)
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
- _Timing:_ Load 1.50s; Gen 3.05s; Total 4.55s
- _Throughput:_ Prompt 10.2 TPS (16 tok); Gen 17.2 TPS (25 tok)
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats lying on a pink blanket, with two remote
> controls placed on the couch behind them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 0.97s; Gen 3.05s; Total 4.03s
- _Throughput:_ Prompt 1,101 TPS (2,356 tok); Gen 62.1 TPS (55 tok)
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
- _Key signals:_ hit token cap (200); reasoning leak
- _Timing:_ Load 2.15s; Gen 3.19s; Total 5.35s
- _Throughput:_ Prompt 1,294 TPS (399 tok); Gen 72.7 TPS (200 tok)
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

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ clean triage pass; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); reasoning leak
- _Timing:_ Load 5.85s; Gen 3.36s; Total 9.21s
- _Throughput:_ Prompt 1,315 TPS (399 tok); Gen 79.8 TPS (200 tok)
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

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 1.61s; Gen 3.79s; Total 5.41s
- _Throughput:_ Prompt 1,295 TPS (1,031 tok); Gen 34.7 TPS (101 tok)
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
- _Timing:_ Load 8.30s; Gen 3.80s; Total 12.10s
- _Throughput:_ Prompt 912 TPS (319 tok); Gen 77.7 TPS (3 tok)
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

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 6.17s; Gen 3.91s; Total 10.10s
- _Throughput:_ Prompt 161 TPS (408 tok); Gen 51.8 TPS (71 tok)
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

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 2.25s; Gen 4.26s; Total 6.52s
- _Throughput:_ Prompt 352 TPS (275 tok); Gen 26.3 TPS (91 tok)
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

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### ✅ mlx-community/Ornith-1.0-35B-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 8.53s; Gen 4.81s; Total 13.36s
- _Throughput:_ Prompt 941 TPS (319 tok); Gen 61.7 TPS (44 tok)
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

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 3.29s; Gen 4.88s; Total 8.18s
- _Throughput:_ Prompt 318 TPS (275 tok); Gen 17.5 TPS (70 tok)
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
  text.; reasoning leak
- _Timing:_ Load 1.68s; Gen 4.89s; Total 6.58s
- _Throughput:_ Prompt 1,403 TPS (1,330 tok); Gen 44.3 TPS (174 tok)
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
- _Timing:_ Load 1.64s; Gen 5.87s; Total 7.52s
- _Throughput:_ Prompt 1,408 TPS (1,239 tok); Gen 40.2 TPS (200 tok)
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
- _Timing:_ Load 1.19s; Gen 6.00s; Total 7.20s
- _Throughput:_ Prompt 152 TPS (745 tok); Gen 51.1 TPS (55 tok)
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
- _Timing:_ Load 1.46s; Gen 6.48s; Total 7.94s
- _Throughput:_ Prompt 2,664 TPS (1,031 tok); Gen 19.6 TPS (119 tok)
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
- _Timing:_ Load 1.74s; Gen 6.84s; Total 8.59s
- _Throughput:_ Prompt 148 TPS (745 tok); Gen 30.4 TPS (54 tok)
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 54 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a red couch. One cat is thin
> with black stripes, while the other is thicker with brown and black stripes.
> Both cats have white bellies and are wearing collars. Two remote controls
> are visible between the cats.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Timing:_ Load 6.17s; Gen 7.49s; Total 13.67s
- _Throughput:_ Prompt 102 TPS (274 tok); Gen 7.55 TPS (36 tok)
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

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); degeneration=incomplete_sentence: ends
  with 'a'
- _Timing:_ Load 8.68s; Gen 9.52s; Total 18.21s
- _Throughput:_ Prompt 68.9 TPS (439 tok); Gen 64.0 TPS (200 tok)
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
- _Timing:_ Load 2.54s; Gen 10.89s; Total 13.44s
- _Throughput:_ Prompt 1,466 TPS (1,239 tok); Gen 19.9 TPS (200 tok)
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
- _Timing:_ Load 3.29s; Gen 12.26s; Total 15.58s
- _Throughput:_ Prompt 277 TPS (319 tok); Gen 14.2 TPS (85 tok)
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
- _Timing:_ Load 2.78s; Gen 17.32s; Total 20.11s
- _Throughput:_ Prompt 8.22 TPS (17 tok); Gen 5.06 TPS (77 tok)
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
- _Timing:_ Load 2.19s; Gen 29.60s; Total 31.81s
- _Throughput:_ Prompt 326 TPS (766 tok); Gen 6.01 TPS (163 tok)
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
- _Timing:_ Load 2.45s; Gen 30.71s; Total 33.17s
- _Throughput:_ Prompt 906 TPS (1,031 tok); Gen 5.39 TPS (159 tok)
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
- _Timing:_ Load 3.34s; Gen 42.60s; Total 45.95s
- _Throughput:_ Prompt 495 TPS (399 tok); Gen 4.81 TPS (200 tok)
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
