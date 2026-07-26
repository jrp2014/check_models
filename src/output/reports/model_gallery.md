# Model Output Gallery

Generated on: 2026-07-26 05:26:15 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Current-run Chooser

Current-run usability and captured resource facts only. Throughput requires at least 16 generated tokens.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability             | Gen TPS             | Peak GB   | Gen tok   | Observations                                                             | Output preview                                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------------------|-----------|-----------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `unusable`            | insufficient sample | 5.5       | 13        | missing requested sections                                               | Two cats sleeping on a pink blanket with two remote controls.                                                                                                                                                                                                                                 |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `unusable`            | 135 tok/s           | 5.2       | 500       | repeated output, token cap truncation                                    | Title: Two cats sleeping on a pink couch with remote controls Description: Two tabby cats are lying on a bright pink couch, one on its back and the other on its side, both appearing to be asleep. A remote control is placed between them on the couch. Keywords: cat, sleeping, co...      |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `unusable`            | 74.4 tok/s          | 16        | 296       | missing requested sections, thinking trace present                       | ◁think▷So, let's analyze the image. First, identify the main elements. There are two tabby cats on a pink fabric surface. One cat is lying on its side, the other is curled up. There are two remote controls (one blue, one silver) near them. The lighting is bright, making the co...      |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `unusable`            | 4.44 tok/s          | 25        | 500       | repeated output, token cap truncation                                    | Title: Two Cats on a Pink Blanket with Remote Controls Description: The image depicts two cats lying on a pink blanket, with two remote controls visible in the background. The cats are positioned on a pink blanket, with one cat on the left and the other on the right. Keywords:...      |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `unusable`            | 54.6 tok/s          | 9.5       | 500       | repeated output, token cap truncation                                    | Title: Two Cats Sleeping on Pink Sofa Description: Two cats are sleeping on a pink sofa, with one cat lying on its side and the other curled up. There are two remote controls on the sofa, one blue and one white. Keywords: cats, sleeping, pink sofa, remote controls, blue, white...      |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 43.8 tok/s          | 14        | 500       | missing requested sections, token cap truncation                         | Here are my reasoning steps: We need to produce a catalog metadata for the image. The image shows two cats sleeping on a pink couch (or pink fabric). There's a remote control on the couch. The cats are tabby with brown, black, and tan stripes. They are lying on their sides, cu...      |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 300 tok/s           | 2.2       | 139       | missing requested sections                                               | Title: "Two Cats Napping on a Pink Sofa" Description: The image shows two cats, one tabby and one striped, both with black stripes and white paws, peacefully sleeping on a pink sofa. The tabby cat is on the left, curled up with its head resting on its paws, while the striped c...      |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `unusable`            | 49.9 tok/s          | 63        | 91        | missing requested sections, unexpected special token                     | <\|begin_of_box\|>Title: Two cats lying on a pink sofa with remote controls Description: Two cats are lying on a pink sofa, one with a green collar and the other without, both appearing to be sleeping. Two remote controls are visible on the sofa between them. Keywords: cats, sle...    |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `unusable`            | 34.5 tok/s          | 19        | 16        | missing requested sections                                               | Title: two tabby cats on a pink blanket with remotes.<end_of_utterance>                                                                                                                                                                                                                       |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `unusable`            | 45.1 tok/s          | 20        | 500       | missing requested sections, token cap truncation, thinking trace present | ◁think▷Okay, let me analyze this image step by step. The image shows two cats lying on a pink fabric surface. Both cats have tabby markings with a mix of brown, black, and gray fur. They appear to be in a relaxed state, possibly sleeping or resting. There are two remote contro...      |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `unusable`            | 333 tok/s           | 3.0       | 261       | missing requested sections                                               | Title: Two cats sleeping on a pink blanket - 5-10 words, concrete and factual; authoritative context may supply identity and location. - Output only the title text after the label. - Do not repeat or paraphrase these instructions in the Title, Description, or Keywords fields....       |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `unusable`            | 15.4 tok/s          | 15        | 500       | repeated output, token cap truncation                                    | Title: Cats Sleeping on a Pink Blanket Description: The image depicts two cats sleeping on a pink blanket, with one cat lying on its back and the other on its side. The cats are both tabby cats with distinctive striped patterns, and they appear to be in a relaxed state, with t...      |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `unusable`            | 56.1 tok/s          | 9.5       | 500       | repeated output, token cap truncation                                    | Title: Two Cats Sleeping on Pink Sofa Description: Two cats are sleeping on a pink sofa, with one cat lying on its side and the other curled up. There are two remote controls on the sofa, one blue and one white. Keywords: cats, sleeping, pink sofa, remote controls, blue, white...      |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `unusable`            | 133 tok/s           | 5.3       | 500       | repeated output, token cap truncation                                    | Title: Two cats sleeping on a pink couch with remote controls Description: Two tabby cats are lying on a bright pink couch, one on its back and the other on its side, both appearing to be asleep. A remote control is placed between them on the couch. Keywords: cat, sleeping, co...      |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | 133 tok/s           | 5.3       | 500       | missing requested sections, token cap truncation                         | Got it, let's tackle this step by step. First, the Title. Need 5-10 words, concrete and factual. The image shows two cats on a pink couch with remotes. So maybe "Two tabby cats resting on pink couch with remotes" but need to be concise. Let's see: "Two tabby cats resting on pi...      |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `unusable`            | insufficient sample | 5.5       | 13        | missing requested sections                                               | Two cats sleeping on a pink blanket with two remote controls.                                                                                                                                                                                                                                 |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | 65.4 tok/s          | 10        | 500       | repeated output, token cap truncation                                    | Title: - Two cats resting on a pink couch Description: - Two tabby cats are curled up on a pink couch, one slightly overlapping the other. A remote control is placed between them. The lighting is soft, highlighting their relaxed postures. Keywords: - Cats, tabby, pink couch, r...      |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `unusable`            | 54.6 tok/s          | 29        | 85        | missing requested sections, unexpected special token                     | <\|channel>thought <channel\|>Title: Two tabby cats sleeping on a pink blanket Description: Two striped tabby cats are curled up on a pink blanket draped over a red sofa. Two remote controls rest on the sofa beside the cats. Keywords: cats, tabby, sleeping, pink, blanket, sofa,...     |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `unusable`            | 59.6 tok/s          | 28        | 80        | missing requested sections, unexpected special token                     | <\|channel>thought <channel\|>Title: Two tabby cats sleeping on a pink blanket Description: Two brown tabby cats rest side by side on a pink blanket over a red sofa. Two television remote controls lie near the cats. Keywords: cats, tabby, sleeping, pink, blanket, red, sofa, remo...    |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `unusable`            | 80.1 tok/s          | 6.0       | 500       | repeated output, missing requested sections, token cap truncation        | - Do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a...      |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 46.9 tok/s          | 17        | 200       | missing requested sections                                               | ## Image Description: Two tabby cats are comfortably asleep on a bright pink sofa. The cat on the left is lying on its side, facing towards the right, with its head resting on its paws. Its tail is curled around its body. The cat on the right is lying on its back, facing towar...      |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `unusable`            | 6.1 tok/s           | 64        | 500       | repeated output, missing requested sections, token cap truncation        | - Do not output any text other than the three sections. - Do not output any text before the three sections. - Do not output any text after the three sections. - Do not output any text between the three sections. - Do not output any text that is not part of the three sections....       |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | 213 tok/s           | 2.3       | 500       | repeated output, missing requested sections, token cap truncation        | Title: 5-10 words, authoritative context may supply identity and location. Description: A close-up of a cat lying on a pink blanket. The cat is wearing a striped shirt and has a green collar. The cat is resting on a pink couch, with a black and white striped blanket underneath...      |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `unusable`            | insufficient sample | 11        | 14        | missing requested sections                                               | - Do not use the word "cat" in the description.                                                                                                                                                                                                                                               |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | 50.2 tok/s          | 4.6       | 500       | repeated output, missing requested sections, token cap truncation        | - Output only the description text after the label. - Output only the keyword list after the label. - Output only the description text after the label. - Output only the keyword list after the label. - Output only the description text after the label. - Output only the keyword...      |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `unusable`            | 37.4 tok/s          | 4.5       | 41        | missing requested sections                                               | Title: Two Striped Cats Sleeping on a Couch Description: Two cats, one striped and the other not, are laying on a couch. The striped cat has a green tag on its ear.                                                                                                                          |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `not_evaluated`       | -                   | -         | -         | none                                                                     | Model loading failed: [Errno 54] Connection reset by peer                                                                                                                                                                                                                                     |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `not_evaluated`       | -                   | -         | -         | none                                                                     | Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.                                                                                                                                                         |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | 517 tok/s           | 1.1       | 81        | none                                                                     | Title: Two cats lounging on a pink couch with remote controls. Description: - 1-2 factual sentences combining supplied authoritative context and clearly visible subjects, setting, colors, composition, and style. - Output only the visible details. Keywords: - 1-2 factual senten...      |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | 31.0 tok/s          | 20        | 75        | none                                                                     | Title: Two cats resting on a pink sofa Description: Two tabby cats lie on a pink sofa, one on its back and the other on its side, with a remote control nearby. Keywords: cats, tabby, sofa, pink, remote control, resting, indoor, domestic, fur, relaxed, lying down, household, pe...      |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `usable_with_caveats` | 63.1 tok/s          | 60        | 462       | unexpected special token                                                 | Alright, let's get to work. I need to analyze this image and generate the metadata. First, I'll break down the image itself. It's a photograph of two cats on a couch. One is a calico, and the other is a tiger tabby. They're both sleeping. The couch is red, and the cats are on...       |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `usable`              | 63.8 tok/s          | 10        | 122       | none                                                                     | Title: Two tabby cats on a pink couch with remote controls. Description: Two tabby cats are lying on a pink couch. One cat is lying on its side with its head resting on the couch, while the other cat is lying on its stomach with its head lowered. Two remote controls are visibl...      |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable`              | 59.1 tok/s          | 7.7       | 500       | none                                                                     | Title: Two tabby cats on a pink couch with remote controls. Description: Two tabby cats are resting on a pink couch. One cat lies on its side with a remote control nearby, while the other cat is positioned with its head down, also near a remote control. The couch has a red upp...      |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `usable`              | 31.7 tok/s          | 19        | 77        | none                                                                     | Title: Two Cats Sleeping on a Sofa Description: Two tabby cats are sleeping on a pink sofa, with two remote controls nearby. The cats are lying on their sides, appearing relaxed and comfortable. Keywords: cats, sleeping, sofa, pink, remote controls, tabby, domestic, relaxation...      |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 34.8 tok/s          | 19        | 62        | none                                                                     | Title: Two cats sleeping on a pink blanket Description: Two tabby cats are lying on a pink blanket with a red couch in the background, accompanied by two remote controls. Keywords: cats, sleeping, pink blanket, red couch, tabby, remote controls, pets, animals, indoor, resting          |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` | 4.06 tok/s          | 39        | 262       | thinking trace present                                                   | ◁think▷So, let's analyze the image. First, identify the elements: two tabby cats on a pink fabric surface, likely a couch. There are two remote controls (one blue, one silver with buttons). The cats are lying down, one facing left, the other facing down. The image has a high-c...      |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | 200 tok/s           | 4.1       | 94        | none                                                                     | Title: - Two cats sleeping on a pink blanket, one with a green collar Description: - Two cats are sleeping on a pink blanket, one with a green collar. - The cats are lying on their sides, with their heads resting on the blanket. - The blanket is spread out on a red couch. - Th...      |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable_with_caveats` | 234 tok/s           | 3.3       | 67        | thinking trace present, thinking trace incomplete                        | &lt;think&gt; Here's the structured output according to your requirements: Title: Two cats resting on a pink surface Description: The image shows two cats lying on a pink fabric. A remote control is visible near one of the cats. Keywords: cats, resting, pink, remote control, tabby,... |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 68.1 tok/s          | 9.8       | 101       | none                                                                     | **Title:** Two cats lounging on pink fabric with remote controls **Description:** A pair of tabby cats, one adult and one juvenile, are lying on their backs on a pink fabric surface, each positioned near a remote control. The setting appears to be indoors with soft, diffused l...      |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              | 65.9 tok/s          | 10        | 106       | none                                                                     | **Title:** Two Tabby Kittens Relaxing on Pink Cushioned Surface **Description:** A pair of tabby kittens lie stretched out on a pink cushioned surface, likely a sofa. Their relaxed postures and proximity to remote controls suggest a domestic, leisurely setting. **Keywords:** t...      |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 199 tok/s           | 4.5       | 100       | none                                                                     | Title: Two resting domestic cats on a pink cushion Description: Two striped domestic cats are lying relaxed on a bright pink cushion, positioned side by side with one slightly behind the other. Their fur contrasts with the fabric, and small remote controls are placed on the cu...      |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable`              | 52.3 tok/s          | 20        | 86        | none                                                                     | Only the three required fields. Title: British cats on red couch Description: Two cats, one grey and black striped, one brown and black striped, lie on a red couch. A pink blanket covers the seat cushions. Two remote controls are visible between the cats. The scene is well-lit...      |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `usable`              | 30.7 tok/s          | 27        | 85        | none                                                                     | Only the three required fields. Title: British cats on red couch Description: Two cats lie on a red couch. One cat wears a green collar. Both cats rest on a pink blanket. Two remote controls lie between the cats. The image is brightly lit, showing the cats' fur patterns clearl...      |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `usable`              | 5.89 tok/s          | 24        | 71        | none                                                                     | Title: Two cats sleeping on pink blanket with remote controls Description: Two cats are resting on a pink blanket, with two remote controls visible behind them. The cats appear to be in a relaxed state, lying on their sides. Keywords: cats, sleeping, pink blanket, remote contr...      |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | 63.0 tok/s          | 71        | 95        | none                                                                     | Title: - Two tabby cats sleeping on a pink sofa with remote controls Description: - Two tabby cats are lying side by side on a bright pink fabric sofa, both appearing to be asleep. A white television remote control is positioned near each cat. Keywords: - cats, tabby, sleeping...      |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable`              | 274 tok/s           | 2.5       | 40        | none                                                                     | Title: Two cats sleeping on a pink blanket Description: Two tabby cats are sleeping on a pink blanket, with one remote control on the couch. Keywords: cats, blanket, remote control                                                                                                          |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `usable`              | 33.1 tok/s          | 19        | 105       | none                                                                     | Title: Two tabby cats sleeping on pink sofa with remotes Description: Two tabby cats are resting on a vibrant pink sofa, one stretched out and the other curled up, with two remote controls placed nearby on the fabric. Keywords: tabby cats, pink sofa, remote controls, sleeping,...      |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `usable`              | 19.1 tok/s          | 30        | 110       | none                                                                     | Title: Two tabby cats sleeping on a pink sofa with remotes Description: Two tabby cats are resting on a bright pink sofa, one lying on its back and the other curled on its side. Two white remote controls are placed on the sofa near the cats. Keywords: tabby cats, sleeping, pin...      |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 77.2 tok/s          | 22        | 126       | none                                                                     | Title: Two tabby cats resting on pink sofa with remotes Description: Two tabby cats lie asleep on a bright pink sofa, one curled near a white remote control and the other stretched out beside a second remote. The scene is brightly lit, highlighting the cats’ striped fur and th...      |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `usable`              | 32.2 tok/s          | 30        | 117       | none                                                                     | Title: Two tabby cats sleeping on a pink sofa with remote controls Description: Two tabby cats are lying asleep on a bright pink sofa, flanking two white remote controls. The scene is lit by natural light, highlighting the striped fur patterns and relaxed posture of the animal...      |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `usable`              | 42.8 tok/s          | 71        | 102       | none                                                                     | Title: Two tabby cats resting on a pink sofa with remote controls Description: Two tabby cats lie asleep on a bright pink sofa, flanking two white remote controls. The scene is lit by bright, even light that highlights the texture of the sofa fabric and the cats' striped fur....       |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 32.3 tok/s          | 7.3       | 104       | none                                                                     | Title: Two tabby cats sleeping on pink sofa with remotes Description: Two tabby cats are lying on a bright pink sofa, each positioned beside a white remote control. The larger cat is curled on its side while the smaller one stretches out lengthwise. Keywords: tabby cats, pink...       |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | 6.38 tok/s          | 30        | 102       | none                                                                     | Title: Two tabby cats sleeping on a pink sofa Description: Two tabby cats are resting on a bright pink sofa, positioned near two white remote controls. The felines are lying on their sides with their eyes closed, appearing to be asleep. Keywords: tabby cats, sleeping, pink sof...      |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | 31.3 tok/s          | 18        | 98        | none                                                                     | Title: Two tabby kittens sleeping on a pink blanket Description: Two tabby kittens are lying asleep on a bright pink blanket, each positioned near a grey television remote control. The kittens exhibit striped markings and appear relaxed in a domestic setting. Keywords: kittens...      |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `usable`              | 17.5 tok/s          | 32        | 105       | none                                                                     | Title: Two tabby cats lying on a pink sofa Description: Two tabby cats are lying on their backs on a bright pink sofa, each with a remote control nearby; both cats appear to be asleep or resting. The cats have striped markings and varying shades of brown and grey fur. Keywords...      |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 113 tok/s           | 16        | 86        | none                                                                     | Title: Two tabby cats sleeping on a pink blanket Description: Two tabby cats lie sleeping on a bright pink blanket next to two remote controls. The scene is captured from a top-down perspective on a red sofa. Keywords: tabby, cats, sleeping, pink, blanket, red, sofa, remote co...      |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 27.6 tok/s          | 20        | 82        | none                                                                     | Title: Two tabby cats sleeping on a pink blanket Description: Two brown tabby cats lie asleep on a bright pink blanket draped over a red sofa. Two remote controls are positioned on the sofa next to the cats. Keywords: cats, tabby, sleeping, pink blanket, red sofa, remote contr...      |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `usable`              | 61.5 tok/s          | 9.7       | 67        | none                                                                     | Title: Two cats sleeping on a pink blanket. Description: Two cats sleeping on a pink blanket. Keywords: Cats, sleeping, pink blanket, cats sleeping, cats on blanket, cats on couch, cats on sofa, cats on red blanket, cats on red couch, cats on red sofa.                                  |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `usable`              | insufficient sample | 12        | 9         | none                                                                     | - Do not output the prompt instructions.                                                                                                                                                                                                                                                      |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `usable`              | insufficient sample | 26        | 9         | none                                                                     | - Do not output the prompt instructions.                                                                                                                                                                                                                                                      |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | 18.6 tok/s          | 15        | 71        | none                                                                     | Title: - Two cats on pink couch Description: - Two tabby cats are resting on a pink couch, with remote controls nearby. Keywords: - cats, tabby, pink, couch, resting, remote controls, relaxed, comfortable, indoor, home, soft, cozy, paws, fur, lounging, peaceful, domestic               |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `usable`              | 7.77 tok/s          | 27        | 71        | none                                                                     | Title: - Two cats on pink couch Description: - Two tabby cats are resting on a pink couch, with remote controls nearby. Keywords: - cats, tabby, pink, couch, resting, remote controls, relaxed, comfortable, indoor, home, soft, cozy, paws, fur, lounging, peaceful, domestic               |
<!-- markdownlint-enable MD034 MD049 -->

## Avoid for This Run

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                        | Usability       | Observations                                                             | Output preview                                                                                                                                                                                                                                                                             |
|--------------------------------------------------------------------------------------------------------------|-----------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                    | `unusable`      | missing requested sections                                               | Two cats sleeping on a pink blanket with two remote controls.                                                                                                                                                                                                                              |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                              | `unusable`      | repeated output, token cap truncation                                    | Title: Two cats sleeping on a pink couch with remote controls Description: Two tabby cats are lying on a bright pink couch, one on its back and the other on its side, both appearing to be asleep. A remote control is placed between them on the couch. Keywords: cat, sleeping, co...   |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                      | `unusable`      | missing requested sections, thinking trace present                       | ◁think▷So, let's analyze the image. First, identify the main elements. There are two tabby cats on a pink fabric surface. One cat is lying on its side, the other is curled up. There are two remote controls (one blue, one silver) near them. The lighting is bright, making the co...   |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                 | `unusable`      | repeated output, token cap truncation                                    | Title: Two Cats on a Pink Blanket with Remote Controls Description: The image depicts two cats lying on a pink blanket, with two remote controls visible in the background. The cats are positioned on a pink blanket, with one cat on the left and the other on the right. Keywords:...   |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                               | `unusable`      | repeated output, token cap truncation                                    | Title: Two Cats Sleeping on Pink Sofa Description: Two cats are sleeping on a pink sofa, with one cat lying on its side and the other curled up. There are two remote controls on the sofa, one blue and one white. Keywords: cats, sleeping, pink sofa, remote controls, blue, white...   |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable`      | missing requested sections, token cap truncation                         | Here are my reasoning steps: We need to produce a catalog metadata for the image. The image shows two cats sleeping on a pink couch (or pink fabric). There's a remote control on the couch. The cats are tabby with brown, black, and tan stripes. They are lying on their sides, cu...   |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable`      | missing requested sections                                               | Title: "Two Cats Napping on a Pink Sofa" Description: The image shows two cats, one tabby and one striped, both with black stripes and white paws, peacefully sleeping on a pink sofa. The tabby cat is on the left, curled up with its head resting on its paws, while the striped c...   |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                         | `unusable`      | missing requested sections, unexpected special token                     | <\|begin_of_box\|>Title: Two cats lying on a pink sofa with remote controls Description: Two cats are lying on a pink sofa, one with a green collar and the other without, both appearing to be sleeping. Two remote controls are visible on the sofa between them. Keywords: cats, sle... |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                      | `unusable`      | missing requested sections                                               | Title: two tabby cats on a pink blanket with remotes.<end_of_utterance>                                                                                                                                                                                                                    |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                  | `unusable`      | missing requested sections, token cap truncation, thinking trace present | ◁think▷Okay, let me analyze this image step by step. The image shows two cats lying on a pink fabric surface. Both cats have tabby markings with a mix of brown, black, and gray fur. They appear to be in a relaxed state, possibly sleeping or resting. There are two remote contro...   |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                   | `unusable`      | missing requested sections                                               | Title: Two cats sleeping on a pink blanket - 5-10 words, concrete and factual; authoritative context may supply identity and location. - Output only the title text after the label. - Do not repeat or paraphrase these instructions in the Title, Description, or Keywords fields....    |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit) | `unusable`      | repeated output, token cap truncation                                    | Title: Cats Sleeping on a Pink Blanket Description: The image depicts two cats sleeping on a pink blanket, with one cat lying on its back and the other on its side. The cats are both tabby cats with distinctive striped patterns, and they appear to be in a relaxed state, with t...   |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)             | `unusable`      | repeated output, token cap truncation                                    | Title: Two Cats Sleeping on Pink Sofa Description: Two cats are sleeping on a pink sofa, with one cat lying on its side and the other curled up. There are two remote controls on the sofa, one blue and one white. Keywords: cats, sleeping, pink sofa, remote controls, blue, white...   |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                  | `unusable`      | repeated output, token cap truncation                                    | Title: Two cats sleeping on a pink couch with remote controls Description: Two tabby cats are lying on a bright pink couch, one on its back and the other on its side, both appearing to be asleep. A remote control is placed between them on the couch. Keywords: cat, sleeping, co...   |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                  | `unusable`      | missing requested sections, token cap truncation                         | Got it, let's tackle this step by step. First, the Title. Need 5-10 words, concrete and factual. The image shows two cats on a pink couch with remotes. So maybe "Two tabby cats resting on pink couch with remotes" but need to be concise. Let's see: "Two tabby cats resting on pi...   |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                          | `unusable`      | missing requested sections                                               | Two cats sleeping on a pink blanket with two remote controls.                                                                                                                                                                                                                              |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `unusable`      | repeated output, token cap truncation                                    | Title: - Two cats resting on a pink couch Description: - Two tabby cats are curled up on a pink couch, one slightly overlapping the other. A remote control is placed between them. The lighting is soft, highlighting their relaxed postures. Keywords: - Cats, tabby, pink couch, r...   |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | `unusable`      | missing requested sections, unexpected special token                     | <\|channel>thought <channel\|>Title: Two tabby cats sleeping on a pink blanket Description: Two striped tabby cats are curled up on a pink blanket draped over a red sofa. Two remote controls rest on the sofa beside the cats. Keywords: cats, tabby, sleeping, pink, blanket, sofa,...  |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | `unusable`      | missing requested sections, unexpected special token                     | <\|channel>thought <channel\|>Title: Two tabby cats sleeping on a pink blanket Description: Two brown tabby cats rest side by side on a pink blanket over a red sofa. Two television remote controls lie near the cats. Keywords: cats, tabby, sleeping, pink, blanket, red, sofa, remo... |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                  | `unusable`      | repeated output, missing requested sections, token cap truncation        | - Do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a...   |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                            | `unusable`      | missing requested sections                                               | ## Image Description: Two tabby cats are comfortably asleep on a bright pink sofa. The cat on the left is lying on its side, facing towards the right, with its head resting on its paws. Its tail is curled around its body. The cat on the right is lying on its back, facing towar...   |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                    | `unusable`      | repeated output, missing requested sections, token cap truncation        | - Do not output any text other than the three sections. - Do not output any text before the three sections. - Do not output any text after the three sections. - Do not output any text between the three sections. - Do not output any text that is not part of the three sections....    |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable`      | repeated output, missing requested sections, token cap truncation        | Title: 5-10 words, authoritative context may supply identity and location. Description: A close-up of a cat lying on a pink blanket. The cat is wearing a striped shirt and has a green collar. The cat is resting on a pink couch, with a black and white striped blanket underneath...   |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)      | `unusable`      | missing requested sections                                               | - Do not use the word "cat" in the description.                                                                                                                                                                                                                                            |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable`      | repeated output, missing requested sections, token cap truncation        | - Output only the description text after the label. - Output only the keyword list after the label. - Output only the description text after the label. - Output only the keyword list after the label. - Output only the description text after the label. - Output only the keyword...   |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                            | `unusable`      | missing requested sections                                               | Title: Two Striped Cats Sleeping on a Couch Description: Two cats, one striped and the other not, are laying on a couch. The striped cat has a green tag on its ear.                                                                                                                       |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                 | `not_evaluated` | none                                                                     | Model loading failed: [Errno 54] Connection reset by peer                                                                                                                                                                                                                                  |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                               | `not_evaluated` | none                                                                     | Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.                                                                                                                                                      |
<!-- markdownlint-enable MD034 MD049 -->

## Lowest-memory Usable Models

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability             |   Peak GB |   Gen tok |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|-----------|-----------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              |       1.1 |        81 |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable`              |       2.5 |        40 |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable_with_caveats` |       3.3 |        67 |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              |       4.1 |        94 |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              |       4.5 |       100 |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              |       7.3 |       104 |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable`              |       7.7 |       500 |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `usable`              |       9.7 |        67 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              |       9.8 |       101 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              |      10   |       106 |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `usable`              |      10   |       122 |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `usable`              |      12   |         9 |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              |      15   |        71 |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              |      16   |        86 |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              |      18   |        98 |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              |      19   |        62 |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `usable`              |      19   |        77 |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `usable`              |      19   |       105 |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              |      20   |        75 |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              |      20   |        82 |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable`              |      20   |        86 |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              |      22   |       126 |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `usable`              |      24   |        71 |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `usable`              |      26   |         9 |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `usable`              |      27   |        71 |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `usable`              |      27   |        85 |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `usable`              |      30   |       110 |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              |      30   |       102 |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `usable`              |      30   |       117 |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `usable`              |      32   |       105 |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` |      39   |       262 |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `usable_with_caveats` |      60   |       462 |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              |      71   |        95 |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `usable`              |      71   |       102 |
<!-- markdownlint-enable MD034 MD049 -->

## Fastest Valid Generation

Fastest valid generation: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 517 tok/s

Average valid generation throughput: 78.7 tok/s

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability             | Gen TPS             |   Gen tok |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------------------|-----------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | 517 tok/s           |        81 |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable`              | 274 tok/s           |        40 |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable_with_caveats` | 234 tok/s           |        67 |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | 200 tok/s           |        94 |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 199 tok/s           |       100 |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 113 tok/s           |        86 |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 77.2 tok/s          |       126 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 68.1 tok/s          |       101 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              | 65.9 tok/s          |       106 |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `usable`              | 63.8 tok/s          |       122 |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `usable_with_caveats` | 63.1 tok/s          |       462 |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | 63.0 tok/s          |        95 |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `usable`              | 61.5 tok/s          |        67 |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable`              | 59.1 tok/s          |       500 |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable`              | 52.3 tok/s          |        86 |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `usable`              | 42.8 tok/s          |       102 |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 34.8 tok/s          |        62 |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `usable`              | 33.1 tok/s          |       105 |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 32.3 tok/s          |       104 |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `usable`              | 32.2 tok/s          |       117 |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `usable`              | 31.7 tok/s          |        77 |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | 31.3 tok/s          |        98 |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | 31.0 tok/s          |        75 |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `usable`              | 30.7 tok/s          |        85 |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 27.6 tok/s          |        82 |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `usable`              | 19.1 tok/s          |       110 |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | 18.6 tok/s          |        71 |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `usable`              | 17.5 tok/s          |       105 |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `usable`              | 7.77 tok/s          |        71 |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | 6.38 tok/s          |       102 |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `usable`              | 5.89 tok/s          |        71 |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` | 4.06 tok/s          |       262 |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `usable`              | insufficient sample |         9 |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `usable`              | insufficient sample |         9 |
<!-- markdownlint-enable MD034 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.8`
- `mlx`: `0.32.1.dev20260725+973e27f82`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.24.0`
- *Python Version:* 3.13.13
- *OS:* Darwin 25.5.0
- *macOS Version:* 26.5.2
- *GPU/Chip:* Apple M5 Max
- *MLX Device:* Apple M5 Max
- *GPU Architecture:* applegpu_g17s
- *RAM:* 128.0 GB
- *Recommended Working Set:* 108 GB
- *Fused Attention:* Available

## Image Metadata

- *Date:* 2026-07-25 19:35:34 BST
- *Time:* 19:35:34

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Analyze this image for cataloguing metadata, using British English.
>
> Describe visible details faithfully. If a visual detail is uncertain,
> ambiguous, partially obscured, or too small to verify, leave it out rather
> than guessing.
>
> No existing catalog metadata is supplied. Base every field only on visual
> evidence in the image.
>
> &#8203;Return exactly these three sections, and nothing else:
>
> &#8203;Title:
> &#45; 5-10 words, concrete and factual; authoritative context may supply
> identity and location.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> &#8203;Description:
> &#45; 1-2 factual sentences combining supplied authoritative context with the
> main visible subject, setting, lighting, action, and distinctive visible
> details.
> &#45; Output only the description text after the label.
>
> &#8203;Keywords:
> &#45; 10-18 unique comma-separated terms covering supplied authoritative context
> and clearly visible subjects, setting, colors, composition, and style.
> &#45; Output only the keyword list after the label.
>
> &#8203;Rules:
> &#45; Include only details that are definitely visible in the image.
> &#45; Do not infer or import metadata that is not visible in the image.
> &#45; Prefer omission to speculation.
> &#45; Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent unless visually obvious.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Complete Per-model Evidence

Complete generated or crash evidence for every attempted model.

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### HuggingFaceTB/SmolVLM-Instruct

<details>
<summary>Complete evidence: HuggingFaceTB/SmolVLM-Instruct</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections
- *Model load time:* 0.73s
- *Generation time:* 0.54s
- *Total time:* 1.28s
- *Input validation time:* 0.00257
- *Prompt preparation time:* 0.00241
- *First-token latency:* 0.391
- *Cleanup time:* 0.0623
- *Prompt tokens:* 1,507
- *Generation tokens:* 13
- *Total tokens:* 1,520
- *Prompt throughput (raw):* 3,849 tok/s
- *Generation throughput (raw):* 132 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.35
- *Model-load active memory:* 4.49
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1499
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

*Complete generated output:*

```text
 Two cats sleeping on a pink blanket with two remote controls.
```

</details>

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### Qwen/Qwen3-VL-2B-Instruct

<details>
<summary>Complete evidence: Qwen/Qwen3-VL-2B-Instruct</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated output, token cap truncation
- *Model load time:* 0.70s
- *Generation time:* 3.84s
- *Total time:* 4.55s
- *Input validation time:* 0.00157
- *Prompt preparation time:* 0.00948
- *First-token latency:* 0.105
- *Cleanup time:* 0.0736
- *Prompt tokens:* 596
- *Generation tokens:* 500
- *Total tokens:* 1,096
- *Prompt throughput (raw):* 5,652 tok/s
- *Generation throughput (raw):* 135 tok/s
- *Peak memory:* 5.2
- *Active memory:* 4.3
- *Cache memory:* 0.16
- *Model-load active memory:* 4.26
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1539
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

*Complete generated output:*

```text
Title:
Two cats sleeping on a pink couch with remote controls

Description:
Two tabby cats are lying on a bright pink couch, one on its back and the other on its side, both appearing to be asleep. A remote control is placed between them on the couch.

Keywords:
cat, sleeping, couch, pink, remote, control, tabby, feline, domestic, pet, furniture, domestication, domestic animal, pet care, pet owner, pet lifestyle, pet companionship, pet sleep, pet comfort, pet relaxation, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet
```

</details>

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### jqlive/Kimi-VL-A3B-Thinking-2506-6bit

<details>
<summary>Complete evidence: jqlive/Kimi-VL-A3B-Thinking-2506-6bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections, thinking trace present
- *Model load time:* 2.16s
- *Generation time:* 4.43s
- *Total time:* 6.60s
- *Input validation time:* 0.00165
- *Prompt preparation time:* 0.00182
- *First-token latency:* 0.362
- *Cleanup time:* 0.0929
- *Prompt tokens:* 678
- *Generation tokens:* 296
- *Total tokens:* 974
- *Prompt throughput (raw):* 1,874 tok/s
- *Generation throughput (raw):* 74.4 tok/s
- *Peak memory:* 16
- *Active memory:* 14
- *Cache memory:* 0.84
- *Model-load active memory:* 13.8
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1654
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

*Complete generated output:*

```text
◁think▷So, let's analyze the image. First, identify the main elements. There are two tabby cats on a pink fabric surface. One cat is lying on its side, the other is curled up. There are two remote controls (one blue, one silver) near them. The lighting is bright, making the colors vivid. The setting is indoors, likely a couch or bed. 

For the title: Need a 5-10 word title. "Two tabby cats resting on pink fabric with remote controls". That's concise and factual.

Description: Two tabby cats rest on a pink fabric surface, with two remote controls nearby. One cat lies on its side, the other curled up. The scene is brightly lit, emphasizing the cats' fur patterns and the vibrant pink background.

Keywords: tabby cats, pink fabric, remote controls, resting, brightly lit, cat fur patterns, indoor setting, two cats, relaxed posture, domestic environment.◁/think▷Title:
Two tabby cats resting on pink fabric with remote controls

Description:
Two tabby cats rest on a pink fabric surface, with two remote controls nearby; one cat lies on its side while the other curls up, and the scene is brightly lit to highlight their fur patterns and the vibrant background.

Keywords:
tabby cats, pink fabric, remote controls, resting, brightly lit, cat fur patterns, indoor setting, two cats, relaxed posture, domestic environment<|im_assistant|>
```

</details>

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### meta-llama/Llama-3.2-11B-Vision-Instruct

<details>
<summary>Complete evidence: meta-llama/Llama-3.2-11B-Vision-Instruct</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated output, token cap truncation
- *Model load time:* 2.33s
- *Generation time:* 114.37s
- *Total time:* 116.71s
- *Input validation time:* 0.00177
- *Prompt preparation time:* 0.00477
- *First-token latency:* 1.83
- *Cleanup time:* 0.0969
- *Prompt tokens:* 295
- *Generation tokens:* 500
- *Total tokens:* 795
- *Prompt throughput (raw):* 161 tok/s
- *Generation throughput (raw):* 4.44 tok/s
- *Peak memory:* 25
- *Active memory:* 21
- *Cache memory:* 3.5
- *Model-load active memory:* 21.3
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1571
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

*Complete generated output:*

```text
Title: Two Cats on a Pink Blanket with Remote Controls

Description: The image depicts two cats lying on a pink blanket, with two remote controls visible in the background. The cats are positioned on a pink blanket, with one cat on the left and the other on the right.

Keywords: Cats, Remote Controls, Pink Blanket, Feline, Domestic, Comfort, Relaxation, Home, Living Room, Couch, Television, Entertainment, Leisure, Pet, Animal, Furry, Whiskers, Stripes, Tabby, Colorful, Patterned, Soft, Cozy, Comfortable, Calm, Peaceful, Quiet, Still, Resting, Sleeping, Lying Down, Horizontal, Top View, Overhead, Above, From Above, Bird's Eye View, Flat, Two, Dual, Pair, Together, Side by Side, Companion, Companionship, Friendship, Affection, Cuddling, Snuggling, Touching, Grooming, Playing, Interactive, Interactive Play, Interactive Playtime, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive
```

</details>

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### microsoft/Phi-3.5-vision-instruct

<details>
<summary>Complete evidence: microsoft/Phi-3.5-vision-instruct</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated output, token cap truncation
- *Model load time:* 0.95s
- *Generation time:* 9.47s
- *Total time:* 10.42s
- *Input validation time:* 0.00157
- *Prompt preparation time:* 0.000934
- *First-token latency:* 0.261
- *Cleanup time:* 0.0768
- *Prompt tokens:* 1,102
- *Generation tokens:* 500
- *Total tokens:* 1,602
- *Prompt throughput (raw):* 4,220 tok/s
- *Generation throughput (raw):* 54.6 tok/s
- *Peak memory:* 9.5
- *Active memory:* 8.3
- *Cache memory:* 0.71
- *Model-load active memory:* 8.3
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1488
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

*Complete generated output:*

```text
Title: Two Cats Sleeping on Pink Sofa

Description: Two cats are sleeping on a pink sofa, with one cat lying on its side and the other curled up. There are two remote controls on the sofa, one blue and one white.

Keywords: cats, sleeping, pink sofa, remote controls, blue, white, curled up, side, curled up, sofa, relaxed, comfortable, indoor, domestic, feline, domestic cat, relaxation, rest, companionship, domesticated, pet, animal, home, furniture, comfort, leisure, domestic life, animal behavior, pet ownership, animal lover, animal care, animal comfort, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welf
```

</details>

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

<details>
<summary>Complete evidence: mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections, token cap truncation
- *Model load time:* 1.78s
- *Generation time:* 12.49s
- *Total time:* 14.29s
- *Input validation time:* 0.00183
- *Prompt preparation time:* 0.0063
- *First-token latency:* 1.07
- *Cleanup time:* 0.0891
- *Prompt tokens:* 1,615
- *Generation tokens:* 500
- *Total tokens:* 2,115
- *Prompt throughput (raw):* 1,503 tok/s
- *Generation throughput (raw):* 43.8 tok/s
- *Peak memory:* 14
- *Active memory:* 13
- *Cache memory:* 0.46
- *Model-load active memory:* 12.6
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1841
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

*Complete generated output:*

```text
Here are my reasoning steps:
We need to produce a catalog metadata for the image. The image shows two cats sleeping on a pink couch (or pink fabric). There's a remote control on the couch. The cats are tabby with brown, black, and tan stripes. They are lying on their sides, curled up, sleeping. The lighting appears to be natural or indoor, soft. The couch is pink. The remote is a typical TV remote with buttons. The cats are close to each other but not touching. The remote is near the left cat. The cats have white paws. The cat on the left is slightly more stretched out, with its head turned to the side. The cat on the right is curled more tightly. The couch appears to be a modern sofa. The remote is a standard rectangular shape with colored buttons. The remote is partially visible, showing some numbers and a "TV" label. The remote is white with blue and black text. The cats have a fluffy coat. The cat on the left has a small green tag on its ear? Actually there is a small green tag on the left cat's ear. The cat on the right has a small blue tag? Actually there is a small blue tag on the right cat's ear. The tags are visible. The cats have a small collar? Not visible. The cat's eyes are closed. The cat's whiskers are visible. The cat's tail is visible. The cat's paws are white with some brown. The cat's fur pattern is typical of a domestic short-haired tabby. The couch is pink, maybe a deep magenta. The remote is placed near the left cat's head. The remote is a typical TV remote with a numeric keypad, volume and channel controls, and a "TV" label. The remote is white with black and blue text. The remote is partially covered by the cat's body. The cat's fur is fluffy. The cat's ears are pointed. The cat's nose is pink. The cat's mouth is closed. The cat's tail is curled. The cat's paws are extended. The cat's eyes are closed. The cat's whiskers are white. The cat's ears have tags. The cat's fur is a mix of brown, black, and tan. The cat's tail is long. The cat's body is curled. The cat's head is turned. The cat's paws are white. The cat's nose
```

</details>

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### mlx-community/FastVLM-0.5B-bf16

<details>
<summary>Complete evidence: mlx-community/FastVLM-0.5B-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections
- *Model load time:* 0.68s
- *Generation time:* 0.58s
- *Total time:* 1.27s
- *Input validation time:* 0.00162
- *Prompt preparation time:* 0.00164
- *First-token latency:* 0.0888
- *Cleanup time:* 0.0771
- *Prompt tokens:* 307
- *Generation tokens:* 139
- *Total tokens:* 446
- *Prompt throughput (raw):* 3,457 tok/s
- *Generation throughput (raw):* 300 tok/s
- *Peak memory:* 2.2
- *Active memory:* 1.2
- *Cache memory:* 0.02
- *Model-load active memory:* 1.25
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1562
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

*Complete generated output:*

```text
Title: "Two Cats Napping on a Pink Sofa"
Description: The image shows two cats, one tabby and one striped, both with black stripes and white paws, peacefully sleeping on a pink sofa. The tabby cat is on the left, curled up with its head resting on its paws, while the striped cat is on the right, also curled up with its head on its paws. There are two remote controls on the sofa, one on the left and one on the right. The image is taken in a well-lit room with a pink blanket or throw on the sofa. The cats appear to be in a relaxed state, enjoying a quiet moment together.
```

</details>

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### mlx-community/GLM-4.6V-nvfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-nvfp4</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections, unexpected special token
- *Model load time:* 5.74s
- *Generation time:* 3.40s
- *Total time:* 9.15s
- *Input validation time:* 0.00163
- *Prompt preparation time:* 0.0069
- *First-token latency:* 1.56
- *Cleanup time:* 0.174
- *Prompt tokens:* 686
- *Generation tokens:* 91
- *Total tokens:* 777
- *Prompt throughput (raw):* 439 tok/s
- *Generation throughput (raw):* 49.9 tok/s
- *Peak memory:* 63
- *Active memory:* 62
- *Cache memory:* 0.21
- *Model-load active memory:* 61.9
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1548
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

*Complete generated output:*

```text
<|begin_of_box|>Title:
Two cats lying on a pink sofa with remote controls

Description:
Two cats are lying on a pink sofa, one with a green collar and the other without, both appearing to be sleeping. Two remote controls are visible on the sofa between them.

Keywords:
cats, sleeping, pink sofa, remote controls, green collar, tabby, striped, feline, domestic, furniture, pets, indoors, relaxed, two animals, cushions<|end_of_box|>
```

</details>

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### mlx-community/Idefics3-8B-Llama3-bf16

<details>
<summary>Complete evidence: mlx-community/Idefics3-8B-Llama3-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections
- *Model load time:* 2.20s
- *Generation time:* 1.58s
- *Total time:* 3.78s
- *Input validation time:* 0.00159
- *Prompt preparation time:* 0.00117
- *First-token latency:* 1.07
- *Cleanup time:* 0.096
- *Prompt tokens:* 2,605
- *Generation tokens:* 16
- *Total tokens:* 2,621
- *Prompt throughput (raw):* 2,440 tok/s
- *Generation throughput (raw):* 34.5 tok/s
- *Peak memory:* 19
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1504
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

*Complete generated output:*

```text
 Title: two tabby cats on a pink blanket with remotes.<end_of_utterance>
```

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### mlx-community/Kimi-VL-A3B-Thinking-8bit

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections, token cap truncation, thinking
  trace present
- *Model load time:* 6.93s
- *Generation time:* 11.54s
- *Total time:* 18.49s
- *Input validation time:* 0.0103
- *Prompt preparation time:* 0.00162
- *First-token latency:* 0.342
- *Cleanup time:* 0.104
- *Prompt tokens:* 678
- *Generation tokens:* 500
- *Total tokens:* 1,178
- *Prompt throughput (raw):* 1,981 tok/s
- *Generation throughput (raw):* 45.1 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.91
- *Model-load active memory:* 17.8
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1654
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

*Complete generated output:*

```text
◁think▷Okay, let me analyze this image step by step. The image shows two cats lying on a pink fabric surface. Both cats have tabby markings with a mix of brown, black, and gray fur. They appear to be in a relaxed state, possibly sleeping or resting. There are two remote controls visible near them, one on the left and another on the right. The lighting seems natural, maybe from a window, creating soft shadows. The image has a warm, slightly overexposed look, which might be due to editing. The background is mostly pink, likely a blanket or bedspread. No other objects or context is visible beyond the immediate area around the cats and remotes. The cats' positions suggest they are close to each other but not interacting directly. The remotes are standard TV remotes, not specialized devices. The image doesn't show any specific brands or unique identifiers. The style is casual, capturing a domestic scene.◁/think▷Title: Two Tabby Cats Resting on Pink Fabric with Remote Controls  
Description: Two tabby cats with brown, black, and gray fur are lying on a pink fabric surface, possibly a blanket or bedspread, with two standard TV remotes positioned near them. The scene is softly lit, likely by natural light, creating gentle shadows.  
Keywords: tabby cats, pink fabric, TV remotes, brown and gray fur, domestic setting, relaxed posture, soft lighting, overexposed lighting, standard remotes, fabric texture, warm tones<|im_assistant|><|im_user|>>=<,<|im_assistant|><|im_user|>, be the state vector, determine the stability of the equilibrium [x1, x2] = X.  
Solution: The eigenvalues are given by λ= 1/2, λ=(−3/2)  
Therefore, the eigenvalues of A are real and distinct, which implies the equilibrium is a saddle point.<|im_assistant|><|im_user|>* = \mathbf{ [0 1; 2 0] }, A = [-5  1  ; 2 -3 ]  
\mathbf{x^T} = [x1, x2]  
x' = -5x1 + x2  
x1' = 2x1 - 3x2  
From the first equation, x2 = x1', substitute into the second equation:  
x1' = -5x1 + x1' = -5x1 + 2x1 -3x
```

</details>

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### mlx-community/LFM2-VL-1.6B-8bit

<details>
<summary>Complete evidence: mlx-community/LFM2-VL-1.6B-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections
- *Model load time:* 0.56s
- *Generation time:* 0.88s
- *Total time:* 1.45s
- *Input validation time:* 0.00159
- *Prompt preparation time:* 0.0012
- *First-token latency:* 0.0782
- *Cleanup time:* 0.0851
- *Prompt tokens:* 566
- *Generation tokens:* 261
- *Total tokens:* 827
- *Prompt throughput (raw):* 7,235 tok/s
- *Generation throughput (raw):* 333 tok/s
- *Peak memory:* 3.0
- *Active memory:* 2.1
- *Cache memory:* 0.03
- *Model-load active memory:* 2.06
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1518
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

*Complete generated output:*

```text
Title: Two cats sleeping on a pink blanket
- 5-10 words, concrete and factual; authoritative context may supply identity and location.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the Title, Description, or Keywords fields.
- No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

Description: The image shows two cats sleeping on a pink blanket. The cats are lying on their sides, with their bodies stretched out. The blanket is a bright pink color, and it appears to be made of a soft, plush material. The cats are both gray and black striped, and they have long, fluffy tails. One cat is lying on its side, while the other is lying on its back. The cats are both sleeping peacefully, with their eyes closed. The blanket is draped over a piece of furniture, and there are two remote controls visible in the image. The remote controls are white and black, and they are lying on the blanket next to the cats. The image is well-lit, and the colors are bright and vibrant. The cats appear to be very relaxed and comfortable, and the overall atmosphere of the image is peaceful and serene.
```

</details>

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

<details>
<summary>Complete evidence: mlx-community/Llama-3.2-11B-Vision-Instruct-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated output, token cap truncation
- *Model load time:* 1.77s
- *Generation time:* 33.91s
- *Total time:* 35.68s
- *Input validation time:* 0.00167
- *Prompt preparation time:* 0.00121
- *First-token latency:* 1.48
- *Cleanup time:* 0.0947
- *Prompt tokens:* 294
- *Generation tokens:* 500
- *Total tokens:* 794
- *Prompt throughput (raw):* 198 tok/s
- *Generation throughput (raw):* 15.4 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.69
- *Model-load active memory:* 11.4
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1571
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

*Complete generated output:*

```text
Title:
Cats Sleeping on a Pink Blanket

Description:
The image depicts two cats sleeping on a pink blanket, with one cat lying on its back and the other on its side. The cats are both tabby cats with distinctive striped patterns, and they appear to be in a relaxed state, with their eyes closed and their paws stretched out. The pink blanket provides a soft and comfortable surface for the cats to sleep on, and it adds a pop of color to the otherwise neutral background.

Keywords:
Tabby cats, sleeping, pink blanket, relaxation, comfort, color, background, neutral, soft, surface, distinctive, patterns, closed eyes, stretched paws, relaxed state, comfortable, cozy, warm, inviting, peaceful, calm, serene, happy, content, playful, curious, mischievous, playful, affectionate, loyal, friendly, social, playful, curious, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful,
```

</details>

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### mlx-community/Phi-3.5-vision-instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Phi-3.5-vision-instruct-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated output, token cap truncation
- *Model load time:* 1.01s
- *Generation time:* 9.23s
- *Total time:* 10.25s
- *Input validation time:* 0.0016
- *Prompt preparation time:* 0.000798
- *First-token latency:* 0.258
- *Cleanup time:* 0.0839
- *Prompt tokens:* 1,102
- *Generation tokens:* 500
- *Total tokens:* 1,602
- *Prompt throughput (raw):* 4,267 tok/s
- *Generation throughput (raw):* 56.1 tok/s
- *Peak memory:* 9.5
- *Active memory:* 8.3
- *Cache memory:* 0.71
- *Model-load active memory:* 8.3
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1488
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

*Complete generated output:*

```text
Title: Two Cats Sleeping on Pink Sofa

Description: Two cats are sleeping on a pink sofa, with one cat lying on its side and the other curled up. There are two remote controls on the sofa, one blue and one white.

Keywords: cats, sleeping, pink sofa, remote controls, blue, white, curled up, side, curled up, sofa, relaxed, comfortable, indoor, domestic, feline, domestic cat, relaxation, rest, companionship, domesticated, pet, animal, home, furniture, comfort, leisure, domestic life, animal behavior, pet ownership, animal lover, animal care, animal comfort, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welf
```

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### mlx-community/Qwen3-VL-2B-Instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Instruct-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated output, token cap truncation
- *Model load time:* 0.95s
- *Generation time:* 3.89s
- *Total time:* 4.85s
- *Input validation time:* 0.00151
- *Prompt preparation time:* 0.00019
- *First-token latency:* 0.103
- *Cleanup time:* 0.0799
- *Prompt tokens:* 596
- *Generation tokens:* 500
- *Total tokens:* 1,096
- *Prompt throughput (raw):* 5,799 tok/s
- *Generation throughput (raw):* 133 tok/s
- *Peak memory:* 5.3
- *Active memory:* 4.3
- *Cache memory:* 0.16
- *Model-load active memory:* 4.27
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1539
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

*Complete generated output:*

```text
Title:
Two cats sleeping on a pink couch with remote controls

Description:
Two tabby cats are lying on a bright pink couch, one on its back and the other on its side, both appearing to be asleep. A remote control is placed between them on the couch.

Keywords:
cat, sleeping, couch, pink, remote, control, tabby, feline, domestic, pet, furniture, domestication, domestic animal, pet care, pet owner, pet lifestyle, pet companionship, pet sleep, pet comfort, pet relaxation, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet
```

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections, token cap truncation
- *Model load time:* 0.77s
- *Generation time:* 3.90s
- *Total time:* 4.68s
- *Input validation time:* 0.00164
- *Prompt preparation time:* 0.011
- *First-token latency:* 0.105
- *Cleanup time:* 0.0811
- *Prompt tokens:* 598
- *Generation tokens:* 500
- *Total tokens:* 1,098
- *Prompt throughput (raw):* 5,705 tok/s
- *Generation throughput (raw):* 133 tok/s
- *Peak memory:* 5.3
- *Active memory:* 4.3
- *Cache memory:* 0.16
- *Model-load active memory:* 4.27
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1547
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

*Complete generated output:*

```text
Got it, let's tackle this step by step. First, the Title. Need 5-10 words, concrete and factual. The image shows two cats on a pink couch with remotes. So maybe "Two tabby cats resting on pink couch with remotes" but need to be concise. Let's see: "Two tabby cats resting on pink couch with remote controls" – that's 10 words, factual.

Description: 1-2 sentences. Combine context with visible details. The setting is a pink couch, two cats (one with green collar, maybe), remotes. So "Two tabby cats rest on a bright pink couch, one with a green collar and the other without, accompanied by two remote controls placed nearby."

Keywords: 10-18 unique terms. Need to list visible elements: tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting, cats, couch, remotes, cats, couch, pink, tabby, resting. Wait, need to pick unique terms. Let's list: "tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting, cats, couch, remotes, cats, couch, pink, tabby, resting" – but need to be unique. Let's check: tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting. Wait, maybe "tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting" – but need to make sure they are unique. Let's see: "tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting" – but maybe better to list: "tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting" – no, need to pick 10-18. Let's think: "tabby cats, pink couch, remote controls, resting, two cats, green collar, couch,
```

</details>

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### mlx-community/SmolVLM-Instruct-bf16

<details>
<summary>Complete evidence: mlx-community/SmolVLM-Instruct-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections
- *Model load time:* 1.25s
- *Generation time:* 0.82s
- *Total time:* 2.08s
- *Input validation time:* 0.00198
- *Prompt preparation time:* 0.00139
- *First-token latency:* 0.581
- *Cleanup time:* 0.104
- *Prompt tokens:* 1,507
- *Generation tokens:* 13
- *Total tokens:* 1,520
- *Prompt throughput (raw):* 2,594 tok/s
- *Generation throughput (raw):* 73.9 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.35
- *Model-load active memory:* 4.51
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1499
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

*Complete generated output:*

```text
 Two cats sleeping on a pink blanket with two remote controls.
```

</details>

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### mlx-community/X-Reasoner-7B-8bit

<details>
<summary>Complete evidence: mlx-community/X-Reasoner-7B-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated output, token cap truncation
- *Model load time:* 1.23s
- *Generation time:* 8.04s
- *Total time:* 9.28s
- *Input validation time:* 0.00203
- *Prompt preparation time:* 0.000447
- *First-token latency:* 0.316
- *Cleanup time:* 0.106
- *Prompt tokens:* 698
- *Generation tokens:* 500
- *Total tokens:* 1,198
- *Prompt throughput (raw):* 2,206 tok/s
- *Generation throughput (raw):* 65.4 tok/s
- *Peak memory:* 10
- *Active memory:* 9.5
- *Cache memory:* 0.09
- *Model-load active memory:* 9.46
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1597
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

*Complete generated output:*

```text
Title:
- Two cats resting on a pink couch

Description:
- Two tabby cats are curled up on a pink couch, one slightly overlapping the other. A remote control is placed between them. The lighting is soft, highlighting their relaxed postures.

Keywords:
- Cats, tabby, pink couch, remote control, resting, curled up, domestic, feline, home, comfort, relaxation, feline behavior, indoor setting, pet, furred, domesticated, feline anatomy, feline behavior, feline comfort, feline rest, feline home, feline remote, feline pink, feline overlap, feline remote placement, feline pink couch, feline rest position, feline remote control, feline home environment, feline pink background, feline remote between, feline curled up, feline pink surface, feline remote control placement, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline
```

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections, unexpected special token
- *Model load time:* 3.30s
- *Generation time:* 2.05s
- *Total time:* 5.38s
- *Input validation time:* 0.00168
- *Prompt preparation time:* 0.0192
- *First-token latency:* 0.351
- *Cleanup time:* 0.151
- *Prompt tokens:* 581
- *Generation tokens:* 85
- *Total tokens:* 666
- *Prompt throughput (raw):* 1,656 tok/s
- *Generation throughput (raw):* 54.6 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1493
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

*Complete generated output:*

```text
<|channel>thought
<channel|>Title:
Two tabby cats sleeping on a pink blanket

Description:
Two striped tabby cats are curled up on a pink blanket draped over a red sofa. Two remote controls rest on the sofa beside the cats.

Keywords:
cats, tabby, sleeping, pink, blanket, sofa, red, remote, control, domestic, striped, brown, grey, fur, resting, resting, indoor
```

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-mxfp8</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections, unexpected special token
- *Model load time:* 3.28s
- *Generation time:* 1.83s
- *Total time:* 5.12s
- *Input validation time:* 0.00164
- *Prompt preparation time:* 0.000284
- *First-token latency:* 0.347
- *Cleanup time:* 0.143
- *Prompt tokens:* 581
- *Generation tokens:* 80
- *Total tokens:* 661
- *Prompt throughput (raw):* 1,674 tok/s
- *Generation throughput (raw):* 59.6 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1493
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

*Complete generated output:*

```text
<|channel>thought
<channel|>Title:
Two tabby cats sleeping on a pink blanket

Description:
Two brown tabby cats rest side by side on a pink blanket over a red sofa. Two television remote controls lie near the cats.

Keywords:
cats, tabby, sleeping, pink, blanket, red, sofa, remote controls, indoors, brown, striped, domestic, resting, domestic, feline
```

</details>

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### mlx-community/gemma-3n-E2B-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E2B-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated output, missing requested sections, token cap
  truncation
- *Model load time:* 1.55s
- *Generation time:* 6.53s
- *Total time:* 8.09s
- *Input validation time:* 0.0019
- *Prompt preparation time:* 0.000191
- *First-token latency:* 0.27
- *Cleanup time:* 0.121
- *Prompt tokens:* 565
- *Generation tokens:* 500
- *Total tokens:* 1,065
- *Prompt throughput (raw):* 2,093 tok/s
- *Generation throughput (raw):* 80.1 tok/s
- *Peak memory:* 6.0
- *Active memory:* 4.5
- *Cache memory:* 0.04
- *Model-load active memory:* 4.48
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1465
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

*Complete generated output:*

```text

- Do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be
```

</details>

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### mlx-community/gemma-3n-E4B-it-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E4B-it-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections
- *Model load time:* 2.39s
- *Generation time:* 4.64s
- *Total time:* 7.05s
- *Input validation time:* 0.00179
- *Prompt preparation time:* 0.0028
- *First-token latency:* 0.364
- *Cleanup time:* 0.127
- *Prompt tokens:* 573
- *Generation tokens:* 200
- *Total tokens:* 773
- *Prompt throughput (raw):* 1,574 tok/s
- *Generation throughput (raw):* 46.9 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1524
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

*Complete generated output:*

```text
## Image Description:

Two tabby cats are comfortably asleep on a bright pink sofa. The cat on the left is lying on its side, facing towards the right, with its head resting on its paws. Its tail is curled around its body. The cat on the right is lying on its back, facing towards the left, with its paws outstretched. Its tail is also curled. 

Between the two cats are two television remotes. One remote is light blue and the other is light green. Both remotes are lying flat on the sofa. The sofa is made of a soft, plush material. The background is out of focus, but appears to be a room with a light-colored wall. 

The overall impression is one of peacefulness and relaxation. The cats appear to be very comfortable and content. 

## Keywords:

cats, tabby, sleeping, sofa, pink, remotes, relaxation, comfort, domestic, pets, home, cozy, peaceful, rest.
```

</details>

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### mlx-community/gemma-4-31b-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-4-31b-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated output, missing requested sections, token cap
  truncation
- *Model load time:* 6.20s
- *Generation time:* 83.96s
- *Total time:* 90.17s
- *Input validation time:* 0.00195
- *Prompt preparation time:* 0.000145
- *First-token latency:* 1.91
- *Cleanup time:* 0.207
- *Prompt tokens:* 573
- *Generation tokens:* 500
- *Total tokens:* 1,073
- *Prompt throughput (raw):* 299 tok/s
- *Generation throughput (raw):* 6.1 tok/s
- *Peak memory:* 64
- *Active memory:* 63
- *Cache memory:* 0.95
- *Model-load active memory:* 62.6
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1456
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

*Complete generated output:*

```text

- Do not output any text other than the three sections.
- Do not output any text before the three sections.
- Do not output any text after the three sections.
- Do not output any text between the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do
```

</details>

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### mlx-community/nanoLLaVA-1.5-4bit

<details>
<summary>Complete evidence: mlx-community/nanoLLaVA-1.5-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated output, missing requested sections, token cap
  truncation
- *Model load time:* 0.52s
- *Generation time:* 2.44s
- *Total time:* 2.96s
- *Input validation time:* 0.00177
- *Prompt preparation time:* 0.00147
- *First-token latency:* 0.0761
- *Cleanup time:* 0.0993
- *Prompt tokens:* 303
- *Generation tokens:* 500
- *Total tokens:* 803
- *Prompt throughput (raw):* 3,983 tok/s
- *Generation throughput (raw):* 213 tok/s
- *Peak memory:* 2.3
- *Active memory:* 0.62
- *Cache memory:* 0.34
- *Model-load active memory:* 0.617
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1553
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

*Complete generated output:*

```text
Title: 5-10 words, authoritative context may supply identity and location.
Description: A close-up of a cat lying on a pink blanket. The cat is wearing a striped shirt and has a green collar. The cat is resting on a pink couch, with a black and white striped blanket underneath. The cat is looking away from the camera, and the background is a pinkish-purple color. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is taken in a room with pink walls and a pink couch. The cat is the only animal in the image, and the cat is the only object in the image. The image is clear and well-lit, with no visible distractions. The cat is the only subject in the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is clear and well-lit, with no visible distractions. The cat is the only subject in the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is clear and well-lit, with no visible distractions. The cat is the only subject in the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only
```

</details>

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### mlx-community/paligemma2-3b-ft-docci-448-bf16

<details>
<summary>Complete evidence: mlx-community/paligemma2-3b-ft-docci-448-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections
- *Model load time:* 1.57s
- *Generation time:* 1.12s
- *Total time:* 2.70s
- *Input validation time:* 0.00172
- *Prompt preparation time:* 0.000112
- *First-token latency:* 0.427
- *Cleanup time:* 0.115
- *Prompt tokens:* 1,330
- *Generation tokens:* 14
- *Total tokens:* 1,344
- *Prompt throughput (raw):* 3,118 tok/s
- *Generation throughput (raw):* 20.9 tok/s
- *Peak memory:* 11
- *Active memory:* 6.1
- *Cache memory:* 3.7
- *Model-load active memory:* 6.09
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1453
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

*Complete generated output:*

```text
- Do not use the word "cat" in the description.
```

</details>

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### mlx-community/paligemma2-3b-pt-896-4bit

<details>
<summary>Complete evidence: mlx-community/paligemma2-3b-pt-896-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated output, missing requested sections, token cap
  truncation
- *Model load time:* 1.36s
- *Generation time:* 11.31s
- *Total time:* 12.69s
- *Input validation time:* 0.00182
- *Prompt preparation time:* 0.000134
- *First-token latency:* 1.14
- *Cleanup time:* 0.16
- *Prompt tokens:* 4,402
- *Generation tokens:* 500
- *Total tokens:* 4,902
- *Prompt throughput (raw):* 3,873 tok/s
- *Generation throughput (raw):* 50.2 tok/s
- *Peak memory:* 4.6
- *Active memory:* 1.7
- *Cache memory:* 1.9
- *Model-load active memory:* 1.74
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1453
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

*Complete generated output:*

```text
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword
```

</details>

---

<a id="model-qnguyen3-nanollava"></a>

### qnguyen3/nanoLLaVA

<details>
<summary>Complete evidence: qnguyen3/nanoLLaVA</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing requested sections
- *Model load time:* 2.04s
- *Generation time:* 1.59s
- *Total time:* 3.64s
- *Input validation time:* 0.0115
- *Prompt preparation time:* 0.00257
- *First-token latency:* 0.392
- *Cleanup time:* 0.361
- *Prompt tokens:* 303
- *Generation tokens:* 41
- *Total tokens:* 344
- *Prompt throughput (raw):* 773 tok/s
- *Generation throughput (raw):* 37.4 tok/s
- *Peak memory:* 4.5
- *Active memory:* 2.1
- *Cache memory:* 0.98
- *Model-load active memory:* 2.12
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1553
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

*Complete generated output:*

```text
Title: Two Striped Cats Sleeping on a Couch
Description: Two cats, one striped and the other not, are laying on a couch. The striped cat has a green tag on its ear.
```

</details>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

<details>
<summary>Complete evidence: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

- *Execution:* indeterminate
- *Usability:* not_evaluated
- *Maintainer status:* none
- *Observations:* none
- *Failure phase:* model_load
- *Error stage:* Network Error
- *Error code:* UNKNOWN_MODEL_LOAD_NETWORK_ERROR
- *Error type:* ValueError
- *Error package:* unknown
- *Error message:* Model loading failed: [Errno 54] Connection reset by peer
- *Root exception type:* ReadError
- *Root exception module:* httpcore
- *Root exception message:* [Errno 54] Connection reset by peer
- *Model load time:* 0.02s
- *Generation time:* -
- *Total time:* 0.03s
- *Input validation time:* 0.00185
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0702
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Stop reason:* exception
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* not captured
- *Image placeholders:* not captured
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* not captured
- *Tokenizer:* not captured
- *Requested model revision:* not requested
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--SmolVLM2-2.2B-Instruct-mlx/snapshots/844516024a1c4400d34489b89ee067d794e432ed
- *Generation settings:* not captured
- *EOS token:* not captured

*Complete traceback:*

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
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 217, in _receive_event
    data = self._network_stream.read(
        self.READ_NUM_BYTES, timeout=timeout
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_backends/sync.py", line 126, in read
    with map_exceptions(exc_map):
         ~~~~~~~~~~~~~~^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ReadError: [Errno 54] Connection reset by peer

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/.worktrees/subtractive-reporting-simplification/src/check_models.py", line 11196, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/.worktrees/subtractive-reporting-simplification/src/check_models.py", line 10581, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/.worktrees/subtractive-reporting-simplification/src/check_models.py", line 701, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 938, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 589, in get_model_path
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
httpx.ReadError: [Errno 54] Connection reset by peer

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/.worktrees/subtractive-reporting-simplification/src/check_models.py", line 11448, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/.worktrees/subtractive-reporting-simplification/src/check_models.py", line 11211, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: [Errno 54] Connection reset by peer

```

*Captured upstream output:*

```text
=== STDERR ===
[04:52:37] DEBUG    HF Cache Info for mlx-community/SmolVLM2-2.2B-Instruct-mlx: size=4290.2 MB,
                    files=12
```

</details>

---

<a id="model-mlx-community-step-37-flash-oq2e"></a>

### mlx-community/Step-3.7-Flash-oQ2e

<details>
<summary>Complete evidence: mlx-community/Step-3.7-Flash-oQ2e</summary>

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Failure phase:* processor_load
- *Error stage:* Processor Error
- *Error code:* MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- *Error type:* ValueError
- *Error package:* model-config
- *Error message:* Model preflight failed for
  mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor;
  expected multimodal processor.
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* Loaded processor has no image_processor; expected
  multimodal processor.
- *Model load time:* 5.52s
- *Generation time:* -
- *Total time:* 6.23s
- *Input validation time:* 0.00159
- *Prompt preparation time:* 1.18e-05
- *First-token latency:* -
- *Cleanup time:* 0.578
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Stop reason:* exception
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* not captured
- *Image placeholders:* not captured
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* not captured
- *Tokenizer:* not captured
- *Requested model revision:* not requested
- *Resolved model revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Step-3.7-Flash-oQ2e/snapshots/3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Generation settings:* not captured
- *EOS token:* not captured

*Complete traceback:*

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/.worktrees/subtractive-reporting-simplification/src/check_models.py", line 10963, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/.worktrees/subtractive-reporting-simplification/src/check_models.py", line 10755, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/.worktrees/subtractive-reporting-simplification/src/check_models.py", line 10688, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/.worktrees/subtractive-reporting-simplification/src/check_models.py", line 11448, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/.worktrees/subtractive-reporting-simplification/src/check_models.py", line 11215, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "~/Documents/AI/mlx/check_models/.worktrees/subtractive-reporting-simplification/src/check_models.py", line 11004, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.

```

*Captured upstream output:*

```text
=== STDERR ===
Fetching 24 files:   0%|          | 0/24 [00:00<?, ?it/s]
Fetching 24 files: 100%|##########| 24/24 [00:00<00:00, 14329.29it/s]
[04:52:43] ERROR    Model preflight validation failed for mlx-community/Step-3.7-Flash-oQ2e
                    ValueError: Loaded processor has no image_processor; expected multimodal
                    processor.
```

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
- *Model load time:* 0.38s
- *Generation time:* 0.19s
- *Total time:* 0.57s
- *Input validation time:* 0.0015
- *Prompt preparation time:* 0.00546
- *First-token latency:* 0.0176
- *Cleanup time:* 0.0592
- *Prompt tokens:* 377
- *Generation tokens:* 81
- *Total tokens:* 458
- *Prompt throughput (raw):* 21,463 tok/s
- *Generation throughput (raw):* 517 tok/s
- *Peak memory:* 1.1
- *Active memory:* 0.90
- *Cache memory:* 0.01
- *Model-load active memory:* 0.898
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1518
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

*Complete generated output:*

```text
Title: Two cats lounging on a pink couch with remote controls.

Description:
- 1-2 factual sentences combining supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.
- Output only the visible details.

Keywords:
- 1-2 factual sentences combining supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.
```

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
- *Model load time:* 2.28s
- *Generation time:* 3.60s
- *Total time:* 5.89s
- *Input validation time:* 0.00153
- *Prompt preparation time:* 0.00659
- *First-token latency:* 1.17
- *Cleanup time:* 0.0997
- *Prompt tokens:* 726
- *Generation tokens:* 75
- *Total tokens:* 801
- *Prompt throughput (raw):* 622 tok/s
- *Generation throughput (raw):* 31.0 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.18
- *Model-load active memory:* 18
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1467
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

*Complete generated output:*

```text
Title:
Two cats resting on a pink sofa

Description:
Two tabby cats lie on a pink sofa, one on its back and the other on its side, with a remote control nearby.

Keywords:
cats, tabby, sofa, pink, remote control, resting, indoor, domestic, fur, relaxed, lying down, household, pet, furniture, soft focus
```

</details>

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected special token
- *Model load time:* 4.82s
- *Generation time:* 8.14s
- *Total time:* 12.97s
- *Input validation time:* 0.00162
- *Prompt preparation time:* 0.00756
- *First-token latency:* 0.813
- *Cleanup time:* 0.153
- *Prompt tokens:* 756
- *Generation tokens:* 462
- *Total tokens:* 1,218
- *Prompt throughput (raw):* 930 tok/s
- *Generation throughput (raw):* 63.1 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.08
- *Model-load active memory:* 58.8
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1657
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

*Complete generated output:*

```text
Alright, let's get to work. I need to analyze this image and generate the metadata. First, I'll break down the image itself. It's a photograph of two cats on a couch. One is a calico, and the other is a tiger tabby. They're both sleeping. The couch is red, and the cats are on a red blanket. The lighting is soft, and the overall mood is peaceful.

Now, for the metadata. The title needs to be concise and factual. I'll go with "Two cats sleeping on a red couch." That's clear and to the point.

Next, the description. I need to combine the key elements: the two cats, their colors, the setting, and the action. I'll write: "A calico cat and a tiger tabby cat are sleeping side-by-side on a red couch covered with a red blanket." That covers everything without adding any unnecessary details.

Finally, the keywords. I need a list of 10-18 terms that accurately describe the image. I'll include the subjects (cat, calico, tiger tabby), the setting (couch, blanket, red), the action (sleeping), and some descriptive terms (fluffy, domestic). I'll make sure to avoid any speculative information. I'll go with: "cat, calico, tiger tabby, couch, blanket, red, sleeping, fluffy, domestic, pet, animal, couch, sofa, red couch, red blanket, two cats, side-by-side."

Okay, I'm ready to output the final metadata.
</think>

Title:
Two cats sleeping on a red couch

Description:
A calico cat and a tiger tabby cat are sleeping side-by-side on a red couch covered with a red blanket. The cats are fluffy and appear to be resting peacefully.

Keywords:
cat, calico, tiger tabby, couch, blanket, red, sleeping, fluffy, domestic, pet, animal, couch, sofa, red couch, red blanket, two cats, side-by-side
```

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
- *Model load time:* 1.76s
- *Generation time:* 2.49s
- *Total time:* 4.27s
- *Input validation time:* 0.0015
- *Prompt preparation time:* 0.0071
- *First-token latency:* 0.562
- *Cleanup time:* 0.0966
- *Prompt tokens:* 686
- *Generation tokens:* 122
- *Total tokens:* 808
- *Prompt throughput (raw):* 1,221 tok/s
- *Generation throughput (raw):* 63.8 tok/s
- *Peak memory:* 10
- *Active memory:* 9.4
- *Cache memory:* 0.06
- *Model-load active memory:* 9.43
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1548
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

*Complete generated output:*

```text
Title:
Two tabby cats on a pink couch with remote controls.

Description:
Two tabby cats are lying on a pink couch. One cat is lying on its side with its head resting on the couch, while the other cat is lying on its stomach with its head lowered. Two remote controls are visible on the couch, one near the head of the first cat and the other near the head of the second cat.

Keywords:
Two tabby cats, pink couch, remote controls, lying cats, domestic cats, couch, furniture, pink fabric, tabby pattern, remote control, pet, animal.
```

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
- *Model load time:* 1.63s
- *Generation time:* 9.00s
- *Total time:* 10.63s
- *Input validation time:* 0.0017
- *Prompt preparation time:* 0.000225
- *First-token latency:* 0.524
- *Cleanup time:* 0.0877
- *Prompt tokens:* 686
- *Generation tokens:* 500
- *Total tokens:* 1,186
- *Prompt throughput (raw):* 1,309 tok/s
- *Generation throughput (raw):* 59.1 tok/s
- *Peak memory:* 7.7
- *Active memory:* 6.8
- *Cache memory:* 0.08
- *Model-load active memory:* 6.78
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1548
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

*Complete generated output:*

```text
Title:
Two tabby cats on a pink couch with remote controls.

Description:
Two tabby cats are resting on a pink couch. One cat lies on its side with a remote control nearby, while the other cat is positioned with its head down, also near a remote control. The couch has a red upper section and a pink lower section.

Keywords:
Two tabby cats, pink couch, remote controls, resting cats, domestic cats, tabby pattern, pink fabric, red couch back, remote control, cat on couch, feline, pet, home setting, indoor scene, pink color, red color, remote control, cat, couch, fabric, pattern, tabby, domestic cat, animal, pet, home, interior, living room, furniture, pink couch, red couch back, remote control, cat on couch, feline, pet, home setting, indoor scene, pink color, red color, remote control, cat, couch, fabric, pattern, tabby, domestic cat, animal, pet, home, interior, living room, furniture, pink couch, red couch back, remote control, cat on couch, feline, pet, home setting, indoor scene, pink color, red color, remote control, cat, couch, fabric, pattern, tabby, domestic cat, animal, pet, home, interior, living room, furniture, pink couch, red couch back, remote control, cat on couch, feline, pet, home setting, indoor scene, pink color, red color, remote control, cat, couch, fabric, pattern, tabby, domestic cat, animal, pet, home, interior, living room, furniture, pink couch, red couch back, remote control, cat on couch, feline, pet, home setting, indoor scene, pink color, red color, remote control, cat, couch, fabric, pattern, tabby, domestic cat, animal, pet, home, interior, living room, furniture, pink couch, red couch back, remote control, cat on couch, feline, pet, home setting, indoor scene, pink color, red color, remote control, cat, couch, fabric, pattern, tabby, domestic cat, animal, pet, home, interior, living room, furniture, pink couch, red couch back, remote control, cat on couch, feline, pet, home setting, indoor scene, pink color, red color, remote control, cat, couch, fabric, pattern,
```

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
- *Model load time:* 1.71s
- *Generation time:* 4.96s
- *Total time:* 6.68s
- *Input validation time:* 0.00163
- *Prompt preparation time:* 0.00166
- *First-token latency:* 2.5
- *Cleanup time:* 0.107
- *Prompt tokens:* 3,622
- *Generation tokens:* 77
- *Total tokens:* 3,699
- *Prompt throughput (raw):* 1,451 tok/s
- *Generation throughput (raw):* 31.7 tok/s
- *Peak memory:* 19
- *Active memory:* 16
- *Cache memory:* 0.83
- *Model-load active memory:* 16.4
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1472
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

*Complete generated output:*

```text
Title: Two Cats Sleeping on a Sofa

Description: Two tabby cats are sleeping on a pink sofa, with two remote controls nearby. The cats are lying on their sides, appearing relaxed and comfortable.

Keywords: cats, sleeping, sofa, pink, remote controls, tabby, domestic, relaxation, home, pets, resting, comfort, living room, striped fur, leisure
```

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
- *Model load time:* 1.67s
- *Generation time:* 3.10s
- *Total time:* 4.77s
- *Input validation time:* 0.00171
- *Prompt preparation time:* 0.000214
- *First-token latency:* 1.29
- *Cleanup time:* 0.102
- *Prompt tokens:* 3,622
- *Generation tokens:* 62
- *Total tokens:* 3,684
- *Prompt throughput (raw):* 2,814 tok/s
- *Generation throughput (raw):* 34.8 tok/s
- *Peak memory:* 19
- *Active memory:* 16
- *Cache memory:* 0.28
- *Model-load active memory:* 15.9
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1472
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

*Complete generated output:*

```text
Title:
Two cats sleeping on a pink blanket

Description:
Two tabby cats are lying on a pink blanket with a red couch in the background, accompanied by two remote controls.

Keywords:
cats, sleeping, pink blanket, red couch, tabby, remote controls, pets, animals, indoor, resting
```

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-2506-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* thinking trace present
- *Model load time:* 3.52s
- *Generation time:* 65.57s
- *Total time:* 69.12s
- *Input validation time:* 0.00175
- *Prompt preparation time:* 0.00158
- *First-token latency:* 0.868
- *Cleanup time:* 0.682
- *Prompt tokens:* 678
- *Generation tokens:* 262
- *Total tokens:* 940
- *Prompt throughput (raw):* 781 tok/s
- *Generation throughput (raw):* 4.06 tok/s
- *Peak memory:* 39
- *Active memory:* 33
- *Cache memory:* 5.1
- *Model-load active memory:* 32.8
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1654
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

*Complete generated output:*

```text
◁think▷So, let's analyze the image. First, identify the elements: two tabby cats on a pink fabric surface, likely a couch. There are two remote controls (one blue, one silver with buttons). The cats are lying down, one facing left, the other facing down. The image has a high-contrast, possibly filtered look. 

Title: "Two tabby cats resting on a pink fabric surface with remote controls nearby"

Description: Two tabby cats lie on a pink fabric surface, with two remote controls positioned near them. The cats display striped fur patterns, and the scene has a stylized, high-contrast visual effect.

Keywords: tabby cats, pink fabric, remote controls, striped fur, high-contrast, resting, couch, domestic setting, feline, fabric texture, electronic devices.◁/think▷Title:
Two tabby cats resting on a pink fabric surface with remote controls nearby

Description:
Two tabby cats lie on a pink fabric surface, with two remote controls positioned near them; the scene has a stylized, high - contrast visual effect.

Keywords:
tabby cats, pink fabric, remote controls, striped fur, high - contrast, resting, couch, domestic setting, feline, fabric texture, electronic devices<|im_assistant|>
```

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
- *Model load time:* 0.57s
- *Generation time:* 0.57s
- *Total time:* 1.14s
- *Input validation time:* 0.00164
- *Prompt preparation time:* 0.0032
- *First-token latency:* 0.0782
- *Cleanup time:* 0.0687
- *Prompt tokens:* 566
- *Generation tokens:* 94
- *Total tokens:* 660
- *Prompt throughput (raw):* 7,242 tok/s
- *Generation throughput (raw):* 200 tok/s
- *Peak memory:* 4.1
- *Active memory:* 3.2
- *Cache memory:* 0.02
- *Model-load active memory:* 3.2
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1518
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

*Complete generated output:*

```text
Title:
- Two cats sleeping on a pink blanket, one with a green collar

Description:
- Two cats are sleeping on a pink blanket, one with a green collar.
- The cats are lying on their sides, with their heads resting on the blanket.
- The blanket is spread out on a red couch.
- The cats appear to be relaxed and comfortable.

Keywords:
- cats, sleeping, pink blanket, red couch, green collar, relaxed, comfortable
```

</details>

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### mlx-community/MiniCPM-V-4.6-8bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.6-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* thinking trace present, thinking trace incomplete
- *Model load time:* 0.96s
- *Generation time:* 0.43s
- *Total time:* 1.40s
- *Input validation time:* 0.00178
- *Prompt preparation time:* 0.00795
- *First-token latency:* 0.113
- *Cleanup time:* 0.0935
- *Prompt tokens:* 529
- *Generation tokens:* 67
- *Total tokens:* 596
- *Prompt throughput (raw):* 4,700 tok/s
- *Generation throughput (raw):* 234 tok/s
- *Peak memory:* 3.3
- *Active memory:* 2.3
- *Cache memory:* 0.05
- *Model-load active memory:* 2.32
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1535
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

*Complete generated output:*

```text
<think>

Here's the structured output according to your requirements:

Title:
Two cats resting on a pink surface

Description:
The image shows two cats lying on a pink fabric. A remote control is visible near one of the cats.

Keywords:
cats, resting, pink, remote control, tabby, sleeping
```

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
- *Model load time:* 1.90s
- *Generation time:* 2.26s
- *Total time:* 4.18s
- *Input validation time:* 0.00159
- *Prompt preparation time:* 0.0064
- *First-token latency:* 0.767
- *Cleanup time:* 0.0999
- *Prompt tokens:* 1,259
- *Generation tokens:* 101
- *Total tokens:* 1,360
- *Prompt throughput (raw):* 1,642 tok/s
- *Generation throughput (raw):* 68.1 tok/s
- *Peak memory:* 9.8
- *Active memory:* 8.0
- *Cache memory:* 0.28
- *Model-load active memory:* 8.01
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3905
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

*Complete generated output:*

```text
**Title:**
Two cats lounging on pink fabric with remote controls

**Description:**
A pair of tabby cats, one adult and one juvenile, are lying on their backs on a pink fabric surface, each positioned near a remote control. The setting appears to be indoors with soft, diffused lighting.

**Keywords:**
tabby cats, lounging, pink fabric, remote controls, indoor setting, soft lighting, relaxed posture, juvenile cat, adult cat, casual composition, pastel background
```

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
- *Model load time:* 1.64s
- *Generation time:* 2.40s
- *Total time:* 4.05s
- *Input validation time:* 0.00161
- *Prompt preparation time:* 0.000209
- *First-token latency:* 0.777
- *Cleanup time:* 0.0945
- *Prompt tokens:* 1,259
- *Generation tokens:* 106
- *Total tokens:* 1,365
- *Prompt throughput (raw):* 1,619 tok/s
- *Generation throughput (raw):* 65.9 tok/s
- *Peak memory:* 10
- *Active memory:* 8.4
- *Cache memory:* 0.28
- *Model-load active memory:* 8.43
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3905
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

*Complete generated output:*

```text
**Title:**
Two Tabby Kittens Relaxing on Pink Cushioned Surface

**Description:**
A pair of tabby kittens lie stretched out on a pink cushioned surface, likely a sofa. Their relaxed postures and proximity to remote controls suggest a domestic, leisurely setting.

**Keywords:**
tabby kittens, pink cushioned sofa, remote controls, domestic setting, relaxed posture, pink fabric, indoor lighting, cushioned surface, close composition, casual scene, soft fur, household environment
```

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
- *Model load time:* 1.15s
- *Generation time:* 0.81s
- *Total time:* 1.97s
- *Input validation time:* 0.00161
- *Prompt preparation time:* 0.00632
- *First-token latency:* 0.296
- *Cleanup time:* 0.0774
- *Prompt tokens:* 1,258
- *Generation tokens:* 100
- *Total tokens:* 1,358
- *Prompt throughput (raw):* 4,254 tok/s
- *Generation throughput (raw):* 199 tok/s
- *Peak memory:* 4.5
- *Active memory:* 2.8
- *Cache memory:* 0.19
- *Model-load active memory:* 2.75
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3904
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

*Complete generated output:*

```text
Title:
Two resting domestic cats on a pink cushion

Description:
Two striped domestic cats are lying relaxed on a bright pink cushion, positioned side by side with one slightly behind the other. Their fur contrasts with the fabric, and small remote controls are placed on the cushion’s surface.

Keywords:
cat, domestic cat, striped tabby, pink cushion, sofa, remote controls, resting, indoor setting, soft lighting, domestic environment, feline companionship, patterned fur, neutral lighting.
```

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
- *Model load time:* 1.36s
- *Generation time:* 5.22s
- *Total time:* 6.59s
- *Input validation time:* 0.00149
- *Prompt preparation time:* 0.000144
- *First-token latency:* 3.56
- *Cleanup time:* 0.0842
- *Prompt tokens:* 1,026
- *Generation tokens:* 86
- *Total tokens:* 1,112
- *Prompt throughput (raw):* 289 tok/s
- *Generation throughput (raw):* 52.3 tok/s
- *Peak memory:* 20
- *Active memory:* 9.0
- *Cache memory:* 0.10
- *Model-load active memory:* 9.04
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1446
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

*Complete generated output:*

```text
 Only the three required fields.

Title:
British cats on red couch

Description:
Two cats, one grey and black striped, one brown and black striped, lie on a red couch. A pink blanket covers the seat cushions. Two remote controls are visible between the cats. The scene is well-lit, suggesting daytime indoor setting.

Keywords:
British, cats, red couch, remote controls, striped, pink blanket, daytime indoor
```

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
- *Model load time:* 1.90s
- *Generation time:* 6.56s
- *Total time:* 8.46s
- *Input validation time:* 0.00145
- *Prompt preparation time:* 0.000113
- *First-token latency:* 3.77
- *Cleanup time:* 0.0837
- *Prompt tokens:* 1,026
- *Generation tokens:* 85
- *Total tokens:* 1,111
- *Prompt throughput (raw):* 272 tok/s
- *Generation throughput (raw):* 30.7 tok/s
- *Peak memory:* 27
- *Active memory:* 16
- *Cache memory:* 0.10
- *Model-load active memory:* 16.1
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1446
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

*Complete generated output:*

```text
 Only the three required fields.

Title:
British cats on red couch

Description:
Two cats lie on a red couch. One cat wears a green collar. Both cats rest on a pink blanket. Two remote controls lie between the cats. The image is brightly lit, showing the cats' fur patterns clearly.

Keywords:
British, cats, red couch, pink blanket, remote controls, green collar, bright lighting, fur patterns
```

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
- *Model load time:* 2.46s
- *Generation time:* 13.15s
- *Total time:* 15.62s
- *Input validation time:* 0.00181
- *Prompt preparation time:* 0.00469
- *First-token latency:* 0.964
- *Cleanup time:* 0.106
- *Prompt tokens:* 1,047
- *Generation tokens:* 71
- *Total tokens:* 1,118
- *Prompt throughput (raw):* 1,086 tok/s
- *Generation throughput (raw):* 5.89 tok/s
- *Peak memory:* 24
- *Active memory:* 19
- *Cache memory:* 4.4
- *Model-load active memory:* 17.4
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1505
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

*Complete generated output:*

```text
Title:
Two cats sleeping on pink blanket with remote controls

Description:
Two cats are resting on a pink blanket, with two remote controls visible behind them. The cats appear to be in a relaxed state, lying on their sides.

Keywords:
cats, sleeping, pink blanket, remote controls, home, indoor, relaxation, feline, comfort, bedding
```

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
- *Model load time:* 6.11s
- *Generation time:* 3.06s
- *Total time:* 9.19s
- *Input validation time:* 0.00181
- *Prompt preparation time:* 0.0137
- *First-token latency:* 1.54
- *Cleanup time:* 0.219
- *Prompt tokens:* 620
- *Generation tokens:* 95
- *Total tokens:* 715
- *Prompt throughput (raw):* 404 tok/s
- *Generation throughput (raw):* 63.0 tok/s
- *Peak memory:* 71
- *Active memory:* 70
- *Cache memory:* 0.10
- *Model-load active memory:* 70.2
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1558
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

*Complete generated output:*

```text
Title:
- Two tabby cats sleeping on a pink sofa with remote controls

Description:
- Two tabby cats are lying side by side on a bright pink fabric sofa, both appearing to be asleep. A white television remote control is positioned near each cat.

Keywords:
- cats, tabby, sleeping, sofa, pink, remote control, furniture, domestic, animals, resting, striped, fur, indoor, pet, couch, relaxation, two
```

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
- *Model load time:* 0.65s
- *Generation time:* 0.33s
- *Total time:* 0.99s
- *Input validation time:* 0.00166
- *Prompt preparation time:* 0.00325
- *First-token latency:* 0.165
- *Cleanup time:* 0.0788
- *Prompt tokens:* 698
- *Generation tokens:* 40
- *Total tokens:* 738
- *Prompt throughput (raw):* 4,220 tok/s
- *Generation throughput (raw):* 274 tok/s
- *Peak memory:* 2.5
- *Active memory:* 1.3
- *Cache memory:* 0.03
- *Model-load active memory:* 1.26
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1597
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

*Complete generated output:*

```text
Title: Two cats sleeping on a pink blanket
Description: Two tabby cats are sleeping on a pink blanket, with one remote control on the couch.
Keywords: cats, blanket, remote control
```

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
- *Model load time:* 2.47s
- *Generation time:* 4.11s
- *Total time:* 6.60s
- *Input validation time:* 0.0016
- *Prompt preparation time:* 0.014
- *First-token latency:* 0.915
- *Cleanup time:* 0.134
- *Prompt tokens:* 620
- *Generation tokens:* 105
- *Total tokens:* 725
- *Prompt throughput (raw):* 678 tok/s
- *Generation throughput (raw):* 33.1 tok/s
- *Peak memory:* 19
- *Active memory:* 18
- *Cache memory:* 0.24
- *Model-load active memory:* 16.1
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1558
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

*Complete generated output:*

```text
Title:
Two tabby cats sleeping on pink sofa with remotes

Description:
Two tabby cats are resting on a vibrant pink sofa, one stretched out and the other curled up, with two remote controls placed nearby on the fabric.

Keywords:
tabby cats, pink sofa, remote controls, sleeping, striped fur, domestic scene, relaxed posture, fabric texture, indoor setting, feline companions, casual arrangement, soft lighting, home environment, pet lifestyle, comfortable seating, animal rest, everyday life
```

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
- *Model load time:* 3.40s
- *Generation time:* 6.76s
- *Total time:* 10.19s
- *Input validation time:* 0.00162
- *Prompt preparation time:* 0.0148
- *First-token latency:* 0.998
- *Cleanup time:* 0.151
- *Prompt tokens:* 620
- *Generation tokens:* 110
- *Total tokens:* 730
- *Prompt throughput (raw):* 621 tok/s
- *Generation throughput (raw):* 19.1 tok/s
- *Peak memory:* 30
- *Active memory:* 29
- *Cache memory:* 0.24
- *Model-load active memory:* 28.7
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1558
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

*Complete generated output:*

```text
Title:
Two tabby cats sleeping on a pink sofa with remotes

Description:
Two tabby cats are resting on a bright pink sofa, one lying on its back and the other curled on its side. Two white remote controls are placed on the sofa near the cats.

Keywords:
tabby cats, sleeping, pink sofa, remote controls, striped fur, domestic animals, resting, bright pink, furniture, indoor, pets, relaxation, striped pattern, soft fabric, two cats, lounging, home interior, animal behaviour
```

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
- *Model load time:* 2.78s
- *Generation time:* 2.06s
- *Total time:* 4.86s
- *Input validation time:* 0.00166
- *Prompt preparation time:* 0.000267
- *First-token latency:* 0.421
- *Cleanup time:* 0.125
- *Prompt tokens:* 620
- *Generation tokens:* 126
- *Total tokens:* 746
- *Prompt throughput (raw):* 1,473 tok/s
- *Generation throughput (raw):* 77.2 tok/s
- *Peak memory:* 22
- *Active memory:* 21
- *Cache memory:* 0.09
- *Model-load active memory:* 20.4
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1558
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

*Complete generated output:*

```text
Title: Two tabby cats resting on pink sofa with remotes

Description: Two tabby cats lie asleep on a bright pink sofa, one curled near a white remote control and the other stretched out beside a second remote. The scene is brightly lit, highlighting the cats’ striped fur and the vivid colour of the upholstery.

Keywords: tabby cats, pink sofa, remote controls, sleeping cats, striped fur, domestic interior, bright lighting, pet relaxation, sofa fabric, animal behaviour, household objects, feline rest, two cats, white remotes, soft furnishings, casual setting, indoor scene, animal companionship
```

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
- *Model load time:* 3.86s
- *Generation time:* 4.19s
- *Total time:* 8.07s
- *Input validation time:* 0.00166
- *Prompt preparation time:* 0.000246
- *First-token latency:* 0.543
- *Cleanup time:* 0.154
- *Prompt tokens:* 620
- *Generation tokens:* 117
- *Total tokens:* 737
- *Prompt throughput (raw):* 1,141 tok/s
- *Generation throughput (raw):* 32.2 tok/s
- *Peak memory:* 30
- *Active memory:* 30
- *Cache memory:* 0.10
- *Model-load active memory:* 29.1
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1558
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

*Complete generated output:*

```text
Title: Two tabby cats sleeping on a pink sofa with remote controls

Description: Two tabby cats are lying asleep on a bright pink sofa, flanking two white remote controls. The scene is lit by natural light, highlighting the striped fur patterns and relaxed posture of the animals.

Keywords: tabby cats, pink sofa, remote controls, sleeping animals, domestic interior, striped fur, feline relaxation, household objects, soft lighting, pet photography, cozy setting, animal behaviour, striped pattern, white remotes, fabric texture, side view, resting pose, indoor scene
```

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
- *Model load time:* 6.60s
- *Generation time:* 3.75s
- *Total time:* 10.37s
- *Input validation time:* 0.0019
- *Prompt preparation time:* 0.000251
- *First-token latency:* 1.35
- *Cleanup time:* 0.394
- *Prompt tokens:* 620
- *Generation tokens:* 102
- *Total tokens:* 722
- *Prompt throughput (raw):* 461 tok/s
- *Generation throughput (raw):* 42.8 tok/s
- *Peak memory:* 71
- *Active memory:* 70
- *Cache memory:* 0.10
- *Model-load active memory:* 70.2
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1558
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

*Complete generated output:*

```text
Title: Two tabby cats resting on a pink sofa with remote controls

Description: Two tabby cats lie asleep on a bright pink sofa, flanking two white remote controls. The scene is lit by bright, even light that highlights the texture of the sofa fabric and the cats' striped fur.

Keywords: cats, tabby, sofa, pink, remote control, sleeping, resting, furniture, domestic, animals, stripes, fur, cushions, living room, leisure, pets, household, relaxation
```

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
- *Model load time:* 2.29s
- *Generation time:* 3.72s
- *Total time:* 6.01s
- *Input validation time:* 0.00288
- *Prompt preparation time:* 0.000295
- *First-token latency:* 0.462
- *Cleanup time:* 0.0964
- *Prompt tokens:* 620
- *Generation tokens:* 104
- *Total tokens:* 724
- *Prompt throughput (raw):* 1,341 tok/s
- *Generation throughput (raw):* 32.3 tok/s
- *Peak memory:* 7.3
- *Active memory:* 6.7
- *Cache memory:* 0.12
- *Model-load active memory:* 5.96
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1558
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

*Complete generated output:*

```text
Title:
Two tabby cats sleeping on pink sofa with remotes

Description:
Two tabby cats are lying on a bright pink sofa, each positioned beside a white remote control. The larger cat is curled on its side while the smaller one stretches out lengthwise.

Keywords:
tabby cats, pink sofa, remote controls, sleeping, striped fur, domestic cats, home interior, soft lighting, relaxed posture, pet comfort, cozy setting, fabric texture, household items, animal rest, casual scene
```

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
- *Model load time:* 3.45s
- *Generation time:* 19.20s
- *Total time:* 22.69s
- *Input validation time:* 0.00156
- *Prompt preparation time:* 0.0156
- *First-token latency:* 3.18
- *Cleanup time:* 0.203
- *Prompt tokens:* 620
- *Generation tokens:* 102
- *Total tokens:* 722
- *Prompt throughput (raw):* 195 tok/s
- *Generation throughput (raw):* 6.38 tok/s
- *Peak memory:* 30
- *Active memory:* 29
- *Cache memory:* 0.24
- *Model-load active memory:* 28.7
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1558
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

*Complete generated output:*

```text
Title:
Two tabby cats sleeping on a pink sofa

Description:
Two tabby cats are resting on a bright pink sofa, positioned near two white remote controls. The felines are lying on their sides with their eyes closed, appearing to be asleep.

Keywords:
tabby cats, sleeping, pink sofa, remote controls, felines, striped fur, domestic animals, rest, relaxation, indoor, pets, lying down, furniture, white, grey, brown, composition, still life
```

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
- *Model load time:* 2.36s
- *Generation time:* 4.12s
- *Total time:* 6.50s
- *Input validation time:* 0.00168
- *Prompt preparation time:* 0.0026
- *First-token latency:* 0.974
- *Cleanup time:* 0.135
- *Prompt tokens:* 574
- *Generation tokens:* 98
- *Total tokens:* 672
- *Prompt throughput (raw):* 589 tok/s
- *Generation throughput (raw):* 31.3 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.44
- *Model-load active memory:* 16.1
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1522
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

*Complete generated output:*

```text
Title:
Two tabby kittens sleeping on a pink blanket

Description:
Two tabby kittens are lying asleep on a bright pink blanket, each positioned near a grey television remote control. The kittens exhibit striped markings and appear relaxed in a domestic setting.

Keywords:
kittens, tabby cats, pink blanket, remote controls, sleeping, domestic animals, striped, grey, relaxed, indoors, feline, pets, close-up, horizontal composition, soft lighting, resting, paws, fur
```

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
- *Model load time:* 3.42s
- *Generation time:* 7.16s
- *Total time:* 10.59s
- *Input validation time:* 0.0016
- *Prompt preparation time:* 0.000218
- *First-token latency:* 1.13
- *Cleanup time:* 0.172
- *Prompt tokens:* 574
- *Generation tokens:* 105
- *Total tokens:* 679
- *Prompt throughput (raw):* 508 tok/s
- *Generation throughput (raw):* 17.5 tok/s
- *Peak memory:* 32
- *Active memory:* 30
- *Cache memory:* 0.44
- *Model-load active memory:* 29.7
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1522
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

*Complete generated output:*

```text
Title:
Two tabby cats lying on a pink sofa

Description:
Two tabby cats are lying on their backs on a bright pink sofa, each with a remote control nearby; both cats appear to be asleep or resting. The cats have striped markings and varying shades of brown and grey fur.

Keywords:
cats, tabby cats, pink, sofa, remote controls, domestic animals, resting, lying down, striped, brown, grey, indoor, close-up, pets, feline, mammals, horizontal, daylight
```

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
- *Model load time:* 2.46s
- *Generation time:* 1.22s
- *Total time:* 3.71s
- *Input validation time:* 0.00165
- *Prompt preparation time:* 0.0204
- *First-token latency:* 0.445
- *Cleanup time:* 0.122
- *Prompt tokens:* 585
- *Generation tokens:* 86
- *Total tokens:* 671
- *Prompt throughput (raw):* 1,316 tok/s
- *Generation throughput (raw):* 113 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.20
- *Model-load active memory:* 15.4
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1521
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

*Complete generated output:*

```text
Title:
Two tabby cats sleeping on a pink blanket

Description:
Two tabby cats lie sleeping on a bright pink blanket next to two remote controls. The scene is captured from a top-down perspective on a red sofa.

Keywords:
tabby, cats, sleeping, pink, blanket, red, sofa, remote control, domestic, animals, pets, top-down, indoor, two, striped, fur
```

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
- *Model load time:* 2.74s
- *Generation time:* 4.07s
- *Total time:* 6.82s
- *Input validation time:* 0.00181
- *Prompt preparation time:* 0.00035
- *First-token latency:* 1.08
- *Cleanup time:* 0.142
- *Prompt tokens:* 585
- *Generation tokens:* 82
- *Total tokens:* 667
- *Prompt throughput (raw):* 541 tok/s
- *Generation throughput (raw):* 27.6 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.77
- *Model-load active memory:* 18.4
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1521
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

*Complete generated output:*

```text
Title: Two tabby cats sleeping on a pink blanket

Description: Two brown tabby cats lie asleep on a bright pink blanket draped over a red sofa. Two remote controls are positioned on the sofa next to the cats.

Keywords: cats, tabby, sleeping, pink blanket, red sofa, remote controls, pets, domestic animals, indoor, brown, pink, red, lying down, fur, home interior
```

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
- *Model load time:* 0.95s
- *Generation time:* 3.10s
- *Total time:* 4.05s
- *Input validation time:* 0.00169
- *Prompt preparation time:* 0.00158
- *First-token latency:* 1.95
- *Cleanup time:* 0.0956
- *Prompt tokens:* 2,685
- *Generation tokens:* 67
- *Total tokens:* 2,752
- *Prompt throughput (raw):* 1,374 tok/s
- *Generation throughput (raw):* 61.5 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.90
- *Model-load active memory:* 8.06
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1469
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

*Complete generated output:*

```text
Title: Two cats sleeping on a pink blanket.

Description: Two cats sleeping on a pink blanket.

Keywords: Cats, sleeping, pink blanket, cats sleeping, cats on blanket, cats on couch, cats on sofa, cats on red blanket, cats on red couch, cats on red sofa. 
```

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
- *Model load time:* 1.72s
- *Generation time:* 1.24s
- *Total time:* 2.97s
- *Input validation time:* 0.00177
- *Prompt preparation time:* 0.000172
- *First-token latency:* 0.96
- *Cleanup time:* 0.119
- *Prompt tokens:* 1,330
- *Generation tokens:* 9
- *Total tokens:* 1,339
- *Prompt throughput (raw):* 1,385 tok/s
- *Generation throughput (raw):* 34.1 tok/s
- *Peak memory:* 12
- *Active memory:* 7.9
- *Cache memory:* 1.7
- *Model-load active memory:* 7.88
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1453
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

*Complete generated output:*

```text
- Do not output the prompt instructions.
```

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
- *Model load time:* 2.62s
- *Generation time:* 2.86s
- *Total time:* 5.49s
- *Input validation time:* 0.00162
- *Prompt preparation time:* 0.000114
- *First-token latency:* 1.3
- *Cleanup time:* 0.139
- *Prompt tokens:* 1,330
- *Generation tokens:* 9
- *Total tokens:* 1,339
- *Prompt throughput (raw):* 1,019 tok/s
- *Generation throughput (raw):* 5.91 tok/s
- *Peak memory:* 26
- *Active memory:* 19
- *Cache memory:* 6.2
- *Model-load active memory:* 19.3
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1453
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

*Complete generated output:*

```text
- Do not output the prompt instructions.
```

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
- *Model load time:* 1.86s
- *Generation time:* 4.84s
- *Total time:* 6.71s
- *Input validation time:* 0.00185
- *Prompt preparation time:* 0.0023
- *First-token latency:* 1.01
- *Cleanup time:* 0.103
- *Prompt tokens:* 1,524
- *Generation tokens:* 71
- *Total tokens:* 1,595
- *Prompt throughput (raw):* 1,510 tok/s
- *Generation throughput (raw):* 18.6 tok/s
- *Peak memory:* 15
- *Active memory:* 14
- *Cache memory:* 0.32
- *Model-load active memory:* 13.5
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1467
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

*Complete generated output:*

```text
Title:
- Two cats on pink couch

Description:
- Two tabby cats are resting on a pink couch, with remote controls nearby.

Keywords:
- cats, tabby, pink, couch, resting, remote controls, relaxed, comfortable, indoor, home, soft, cozy, paws, fur, lounging, peaceful, domestic
```

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
- *Model load time:* 2.76s
- *Generation time:* 12.70s
- *Total time:* 15.48s
- *Input validation time:* 0.00159
- *Prompt preparation time:* 0.000204
- *First-token latency:* 3.53
- *Cleanup time:* 0.671
- *Prompt tokens:* 1,524
- *Generation tokens:* 71
- *Total tokens:* 1,595
- *Prompt throughput (raw):* 431 tok/s
- *Generation throughput (raw):* 7.77 tok/s
- *Peak memory:* 27
- *Active memory:* 25
- *Cache memory:* 0.31
- *Model-load active memory:* 25.4
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1467
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

*Complete generated output:*

```text
Title:
- Two cats on pink couch

Description:
- Two tabby cats are resting on a pink couch, with remote controls nearby.

Keywords:
- cats, tabby, pink, couch, resting, remote controls, relaxed, comfortable, indoor, home, soft, cozy, paws, fur, lounging, peaceful, domestic
```

</details>

---
