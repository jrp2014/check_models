# Model Output Gallery

Generated on: 2026-08-02 01:13:52 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability       | Total s   | Gen TPS             | Peak GB   | Gen tok   | Observations                                                                                                                                                                                                        | Output preview                                                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------|-----------|---------------------|-----------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`        | 1.30s     | 511 tok/s           | 1.3       | 95        | none                                                                                                                                                                                                                | Title:<br>The Cut in Halesworth<br><br>Description:<br>A large brick industrial building stands prominently in the town centre of Halesworth, England, with a clear blue sky and fluffy clouds overhead.<br><br>Keywords:<br>Industrial building, Halesworth, England, Brickwork, Sky, Clouds, Brick, Indu... |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable`        | 16.97s    | 93.9 tok/s          | 8.6       | 110       | none                                                                                                                                                                                                                | Title:<br>The Cut, Halesworth, Suffolk<br><br>Description:<br>A red brick industrial building in Halesworth, Suffolk, UK, featuring a prominent white section with "THE CUT" sign, under a partly cloudy blue sky. The structure has a gravel foreground and is surrounded by greenery and a low...           |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `usable`        | 28.23s    | 5.02 tok/s          | 25        | 114       | none                                                                                                                                                                                                                | Title: Red Brick Building with The Cut Sign in Halesworth, England<br><br>Description: A large red brick building with a distinctive roof and a prominent sign reading "The Cut" stands in Halesworth, England, under a blue sky with clouds, with a gravel area in front and a wall surrou...                |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `usable`        | 3.71s     | 55.8 tok/s          | 9.4       | 94        | none                                                                                                                                                                                                                | Title: Halesworth Town Centre, Suffolk, England<br><br>Description: The Cut, a prominent red brick building in Halesworth, Suffolk, England, stands as a historical industrial landmark with a distinctive white sign on its facade.<br><br>Keywords: Halesworth, Suffolk, England, red brick bui...          |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`        | 10.60s    | 27.1 tok/s          | 23        | 109       | none                                                                                                                                                                                                                | Title:<br>The Cut Arts Centre, Halesworth<br><br>Description:<br>A red-brick industrial building with white-framed windows, known as The Cut Arts Centre, stands under a partly cloudy sky in Halesworth, Suffolk, UK.<br><br>Keywords:<br>The Cut Arts Centre, Halesworth, Suffolk, UK, red brick buildin... |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable`        | 11.94s    | 75.4 tok/s          | 8.4       | 110       | none                                                                                                                                                                                                                | Title: The Cut, Halesworth<br>Description: The image shows "The Cut" building, a large red brick industrial-style structure with a white section displaying "THE CUT" signage, located in Halesworth, Suffolk, England. The scene includes a gravel foreground, a brick wall, and greene...                   |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable`        | 7.99s     | 32.4 tok/s          | 19        | 122       | none                                                                                                                                                                                                                | Title: The Cut in Halesworth, Suffolk, England, UK.<br>Description: The Cut is a historic brick building in Halesworth, Suffolk, England, UK, featuring a red brick facade with multiple windows and a white sign with the text "The Cut" on it. The building is surrounded by a brick w...                   |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `usable`        | 7.10s     | 32.5 tok/s          | 18        | 89        | none                                                                                                                                                                                                                | Title:<br>Red Brick Building, Halesworth, Suffolk<br><br>Description:<br>A red brick building in Halesworth, Suffolk, UK, with "The Cut" signage on the side, under a blue sky with scattered clouds.<br><br>Keywords:<br>Adobe Stock, Arts centre, Brickwork, Bushes, England, Europe, Gravel, Halesworth... |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`        | 5.93s     | 34.4 tok/s          | 18        | 84        | none                                                                                                                                                                                                                | Title: The Cut in Halesworth, Suffolk<br>Description: A red brick building with "The Cut" sign stands under a blue sky with clouds, located in Halesworth, Suffolk, UK.<br>Keywords: Red Brick Building, Blue Sky, Clouds, The Cut, Halesworth, Suffolk, England, Europe, Brickwork, Sign,...                 |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`        | 2.25s     | 187 tok/s           | 4.1       | 127       | none                                                                                                                                                                                                                | Title: The Cut, Halesworth Mill, Suffolk, England<br><br>Description: A historic red brick industrial building stands prominently in the town centre of Halesworth, Suffolk, England. The structure, known as The Cut, features a distinctive roof and a sign indicating its industrial pas...                |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`        | 8.82s     | 21.7 tok/s          | 15        | 105       | none                                                                                                                                                                                                                | Title: Red Brick Building with The Cut Sign in Halesworth, England<br><br>Description: A large, red brick building with a distinctive roof and a prominent sign reading "The Cut" stands in Halesworth, England, on a sunny day with a blue sky and white clouds. The building's industrial...                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`        | 6.42s     | 67.0 tok/s          | 13        | 140       | none                                                                                                                                                                                                                | Title:<br>**"The Cut – Historic Brick Mill, Halesworth, Suffolk"**<br><br>Description:<br>A repurposed red-brick industrial mill, now housing **The Cut** arts centre, stands under a bright blue sky with scattered clouds. The three-storey structure features arched windows and weathered bri...          |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`        | 7.30s     | 63.9 tok/s          | 13        | 187       | none                                                                                                                                                                                                                | **Title:**<br>Halesworth’s Historic Brick Mill – *THE CUT* Arts Centre<br><br>**Description:**<br>A large, multi-storey red-brick mill building photographed on **1 August 2026** near **Halesworth, Suffolk, UK (GPS: 52.345200°N, 1.503700°E)**, now housing *THE CUT* arts centre. Its industr...          |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`        | 3.61s     | 187 tok/s           | 7.3       | 141       | none                                                                                                                                                                                                                | Title:<br>**The Cut Industrial Mill in Halesworth, Suffolk**<br><br>Description:<br>A restored red-brick industrial mill in Halesworth, Suffolk, UK, now repurposed as *The Cut* arts centre, bathed in bright blue sky lighting with scattered clouds. The building’s two-storey brick facade, l...          |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`        | 69.52s    | 59.8 tok/s          | 76        | 126       | none                                                                                                                                                                                                                | Title: The Cut Arts Centre, Halesworth, Suffolk<br>Description: A large red-brick industrial building housing The Cut arts centre in Halesworth, Suffolk, England, featuring a prominent white mural with the venue's name and website. The scene is captured under a blue sky with scat...                   |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`        | 4.19s     | 53.7 tok/s          | 9.4       | 94        | none                                                                                                                                                                                                                | Title: Halesworth Town Centre, Suffolk, England<br><br>Description: The Cut, a prominent red brick building in Halesworth, Suffolk, England, stands as a historical industrial landmark with a distinctive white sign on its facade.<br><br>Keywords: Halesworth, Suffolk, England, red brick bui...          |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable`        | 23.92s    | 89.4 tok/s          | 8.6       | 110       | none                                                                                                                                                                                                                | Title:<br>The Cut, Halesworth, Suffolk<br><br>Description:<br>A red brick industrial building in Halesworth, Suffolk, UK, featuring a prominent white section with "THE CUT" sign, under a partly cloudy blue sky. The structure has a gravel foreground and is surrounded by greenery and a low...           |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `usable`        | 102.09s   | 22.6 tok/s          | 26        | 121       | none                                                                                                                                                                                                                | Title: The Cut Arts Centre, Halesworth, Suffolk<br><br>Description: A large red brick industrial building, identified as The Cut arts centre, stands behind a weathered brick wall under a blue sky with scattered clouds. The structure features multiple rows of sash windows and a moder...                |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `usable`        | 115.07s   | 14.9 tok/s          | 38        | 131       | none                                                                                                                                                                                                                | Title:<br>The Cut Arts Centre, Halesworth, Suffolk<br><br>Description:<br>A large, multi-storey red brick industrial building, identified by a white sign as "The Cut," stands behind a weathered brick and concrete wall under a blue sky with scattered clouds. The structure features rows of...           |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`        | 77.30s    | 107 tok/s           | 26        | 122       | none                                                                                                                                                                                                                | Title: The Cut Arts Centre Halesworth Red Brick Building<br><br>Description: A large red brick industrial building converted into The Cut arts centre, featuring a white painted gable end with signage, set against a blue sky with scattered clouds. The structure is situated behind a w...                |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `usable`        | 77.61s    | 65.0 tok/s          | 35        | 114       | none                                                                                                                                                                                                                | Title: The Cut Arts Centre Halesworth Red Brick Building<br><br>Description: A large red brick industrial building converted into an arts centre, featuring a prominent white sign reading "THE CUT" on its side. The structure stands behind a brick wall under a blue sky with scattered...                 |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `usable`        | 94.78s    | 65.6 tok/s          | 76        | 110       | none                                                                                                                                                                                                                | Title: The Cut Arts Centre Halesworth Red Brick Building<br><br>Description: A large red brick industrial building converted into an arts centre, featuring a prominent white sign reading "THE CUT" against a blue sky with scattered clouds. The structure stands behind a brick wall wit...                |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`        | 73.28s    | 90.7 tok/s          | 11        | 109       | none                                                                                                                                                                                                                | Title: The Cut Arts Centre, Halesworth, Suffolk, UK<br><br>Description: A red brick industrial building with a white sign reading "THE CUT" and website URL, set against a blue sky with scattered clouds, located in Halesworth, Suffolk, England.<br><br>Keywords: Adobe Stock, Any Vision, Art...          |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`        | 117.12s   | 14.2 tok/s          | 38        | 120       | none                                                                                                                                                                                                                | Title:<br>The Cut Arts Centre in Halesworth, Suffolk<br><br>Description:<br>A large, historic red brick industrial building, identified as The Cut arts centre, stands behind a weathered brick wall and concrete barrier. The scene is set under a bright blue sky with scattered white clouds,...           |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`        | 9.00s     | 29.5 tok/s          | 18        | 132       | none                                                                                                                                                                                                                | Title: The Cut, Halesworth: Red Brick Arts Centre, Suffolk<br>Description: Captured on 1st August 2026, this image shows The Cut, a large red-brick building in Halesworth, Suffolk, under a clear blue sky; a sign displays the venue’s website address. A gravelled area and brick wal...                   |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `usable`        | 12.25s    | 17.1 tok/s          | 32        | 116       | none                                                                                                                                                                                                                | Title: The Cut, Halesworth: Red Brick Arts Centre, Suffolk<br>Description: Captured on 1st August 2026, The Cut arts centre stands in Halesworth, Suffolk, bathed in bright sunlight. The large red brick building features a distinctive gabled roof and a prominent sign displaying it...                   |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`        | 4.82s     | 120 tok/s           | 16        | 105       | none                                                                                                                                                                                                                | Title: Red brick building The Cut in Halesworth, Suffolk<br><br>Description: A large, historic red brick industrial building known as The Cut stands under a blue sky with white clouds in Halesworth, Suffolk. The multi-storey structure features various window styles and a large white...                |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`        | 8.86s     | 25.9 tok/s          | 20        | 102       | none                                                                                                                                                                                                                | Title: The Cut Arts Centre in Halesworth, Suffolk<br>Description: A large, multi-storey red brick industrial mill building, now The Cut arts centre, under a blue sky with scattered clouds in Halesworth, Suffolk. The structure features numerous small windows and a large white sign...                   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`        | 7.69s     | 36.9 tok/s          | 15        | 96        | none                                                                                                                                                                                                                | Title:<br>Red Brick Mill Building in Halesworth, Suffolk<br><br>Description:<br>A red brick mill building in Halesworth, Suffolk, stands prominently under a blue sky with scattered clouds, featuring the sign "The Cut" on its facade.<br><br>Keywords:<br>Red Brick Building, Halesworth, Suffolk, Mill... |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `usable`        | 10.82s    | 19.8 tok/s          | 27        | 94        | none                                                                                                                                                                                                                | Title:<br>Red Brick Mill Building in Halesworth, Suffolk<br><br>Description:<br>A red brick mill building in Halesworth, Suffolk, stands prominently under a blue sky with scattered clouds, featuring a sign that reads "The Cut."<br><br>Keywords:<br>Red Brick Building, Halesworth, Suffolk, Mill, Ind... |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `unusable`      | 13.73s    | insufficient sample | 5.5       | 13        | Required fields are missing or empty                                                                                                                                                                                | The Cut in Halesworth, Suffolk in the UK.                                                                                                                                                                                                                                                                     |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `unusable`      | 10.64s    | 72.6 tok/s          | 18        | 500       | Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible  | ◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.<br><br>The image shows a large red-brick building, which is "The Cut" in Halesworth, Suffolk,...                |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`      | 16.54s    | 41.4 tok/s          | 15        | 500       | Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | Here are my reasoning steps:<br>We need to produce British-English catalogue metadata from the image and supplied context. The image shows a large brick building, likely a mill, with a sign "THE CUT" on a white-painted side. The building is red brick, multiple stories, with many...                    |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`      | 27.72s    | 52.4 tok/s          | 60        | 500       | Response repeats the same text; Unrecognised model control tokens remain visible; Extra text appears before the Title field; Response appears cut off at the token limit                                            | Alright, let's get to work. I've got the image and the context, and I need to generate some metadata. First, the title. The hints are a bit of a mess, but the image is clear. "The Cut, Halesworth, Suffolk" is the most accurate and concise. Done.<br><br>Next, the description. I need...                 |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`      | 3.25s     | 331 tok/s           | 2.1       | 124       | Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                     | A concrete 5-10-word title: Halesworth Industrial Site<br>A 1-2-sentence factual description: The Cut in Halesworth, Suffolk, UK, GBR, Europe, 2026-08-01 16:28:40 UTC+01:00<br>Keywords: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe,...                 |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `unusable`      | 14.76s    | 52.5 tok/s          | 11        | 106       | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   | <\|begin_of_box\|>Title: The Cut, Halesworth<br>Description: The image shows "The Cut" arts centre, a large red brick industrial-style building with a light blue section displaying the name and website, situated behind a brick wall with overgrown vegetation, under a blue sky with c...                 |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `unusable`      | 36.12s    | 36.5 tok/s          | 78        | 104       | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   | <\|begin_of_box\|>Title: The Cut, Halesworth, Suffolk, England<br>Description: A red brick industrial building with a light blue section displaying "THE CUT" signage, set against a blue sky with clouds, surrounded by brick walls and gravel, located in Halesworth, Suffolk, UK.<br>Keywo...              |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`      | 113.41s   | 4.65 tok/s          | 40        | 500       | Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible  | ◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a title, a descriptive paragraph, and 10-18 keywords.<br><br>Starting with the title: The image shows a large red-brick building w...                |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `unusable`      | 11.90s    | 65.8 tok/s          | 22        | 500       | Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible  | ◁think▷Okay, let's tackle this query. The user wants me to create British-English catalogue metadata based on the image and the provided context. First, I need to parse the instructions carefully. The key points are:<br><br>1. **Title**: A concrete 5-10 word title.<br>2. **Description*...             |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `unusable`      | 3.16s     | 327 tok/s           | 3.0       | 500       | Response repeats the same text; Response appears cut off at the token limit                                                                                                                                         | Title: The Cut, Halesworth, Suffolk, England, UK, GBR, Europe<br><br>Description: A red brick building with a white sign reading "The Cut" stands prominently in the town center of Halesworth, Suffolk, England. The building, likely an industrial or historical structure, features a se...                |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`      | 2.86s     | 272 tok/s           | 3.8       | 55        | Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                     | Industrial brick building with modern sign in town center.<br><br>Keywords: brickwork, industrial, mill, red brick, building, sign, sky, clouds, England, Europe, Suffolk, Gravel, bushes, roof, trees, Car, Clouds, England, Europe                                                                          |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `unusable`      | 20.55s    | insufficient sample | 41        | 1         | No response text was returned; Required fields are missing or empty                                                                                                                                                 | empty output                                                                                                                                                                                                                                                                                                  |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `unusable`      | 20.58s    | insufficient sample | 48        | 1         | No response text was returned; Required fields are missing or empty                                                                                                                                                 | empty output                                                                                                                                                                                                                                                                                                  |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`      | 25.38s    | 6 tok/s             | 26        | 115       | Required fields are missing or empty                                                                                                                                                                                | The Cut Arts Centre, Halesworth, Suffolk, UK<br><br>The Cut arts centre stands in Halesworth, Suffolk, UK, featuring a red brick building with a white side and a sign reading "The Cut" and "www.the-cut.org.uk". The structure is surrounded by a brick wall with bushes and trees, set a...                |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `unusable`      | 80.53s    | 200 tok/s           | 5.1       | 128       | Required fields are missing or empty                                                                                                                                                                                | Title:<br>The Cut, Halesworth, Suffolk, UK, GBR, Europe<br><br>Description:<br>The Cut is a town centre arts centre located in Halesworth, Suffolk, UK. The building is a large, multi-story brick structure with a prominent sign that reads "The Cut" and a website address. The building is su...          |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`      | 29.29s    | 78.8 tok/s          | 8.6       | 500       | Required fields are missing or empty; Response appears cut off at the token limit                                                                                                                                   | Got it, let's tackle this step by step. First, the title. The context says "The Cut in Halesworth, Suffolk in the UK" and the image shows a red brick building with "THE CUT" sign. So the title should be concise. Maybe "The Cut, Halesworth, Suffolk, UK" but need to check if it'...                      |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `unusable`      | 2.97s     | insufficient sample | 5.5       | 13        | Required fields are missing or empty                                                                                                                                                                                | The Cut in Halesworth, Suffolk in the UK.                                                                                                                                                                                                                                                                     |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`      | 44.18s    | 47.5 tok/s          | 14        | 500       | Response repeats the same text; Response appears cut off at the token limit                                                                                                                                         | Title: The Cut in Halesworth, Suffolk<br><br>Description: A historic red brick building with "The Cut" signage stands under a bright blue sky, showcasing industrial architecture in Halesworth, Suffolk, UK.<br><br>Keywords: The Cut, Halesworth, Suffolk, England, UK, Europe, Red brick build...          |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `unusable`      | 7.22s     | 51.9 tok/s          | 29        | 85        | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   | <\|channel>thought<br><channel\|>Title: The Cut Arts Centre in Halesworth, Suffolk<br>Description: A large multi-story red brick industrial building repurposed as an arts centre stands in Halesworth, England, under a partly cloudy sky.<br>Keywords: Halesworth, Suffolk, England, red brick...           |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `unusable`      | 6.31s     | 70.9 tok/s          | 28        | 86        | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   | <\|channel>thought<br><channel\|>Title: The Cut Arts Centre Building in Halesworth, Suffolk<br>Description: A large multi-story red brick industrial building serves as The Cut arts centre, located in Halesworth under a partly cloudy sky.<br>Keywords: Halesworth, Suffolk, England, industr...           |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `unusable`      | 6.63s     | 124 tok/s           | 6.0       | 500       | Required fields are missing or empty; Response appears cut off at the token limit                                                                                                                                   | <br>- 10-18-10-18-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-...                   |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`      | 5.74s     | 46.6 tok/s          | 17        | 104       | Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                     | The image shows a large, red brick building in Halesworth, Suffolk, UK. The building has multiple windows and a white sign that reads "THE CUT" in large letters. The building is surrounded by brick walls and some greenery. The sky is blue with some white clouds.<br><br>**Keywords:**...                |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `unusable`      | 20.53s    | insufficient sample | 64        | 1         | No response text was returned; Required fields are missing or empty                                                                                                                                                 | empty output                                                                                                                                                                                                                                                                                                  |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`      | 7.11s     | 57.3 tok/s          | 9.7       | 133       | Required fields are missing or empty                                                                                                                                                                                | The Cut, Halesworth, Suffolk, England, UK, GBR, Europe, 2026-08-01 16:28:40 UTC+01:00, 52.345200°N, 1.503700°E, Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building,...                      |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`      | 1.98s     | 345 tok/s           | 2.3       | 121       | Required fields are missing or empty                                                                                                                                                                                | Title: The Cut in Halesworth, England, UK, GBR, Europe<br>Description: The Brickwork in Halesworth, Suffolk in the UK, featuring the Cut in the town center. The image captures the essence of the town's historic architecture, with the red brick building standing out amidst the gre...                   |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `unusable`      | 4.33s     | insufficient sample | 12        | 12        | Required fields are missing or empty                                                                                                                                                                                | Write the text in the same font as the image.                                                                                                                                                                                                                                                                 |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `unusable`      | 7.08s     | insufficient sample | 26        | 10        | Required fields are missing or empty                                                                                                                                                                                | Write the text in a plain text editor.                                                                                                                                                                                                                                                                        |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `unusable`      | 11.32s    | 18.6 tok/s          | 11        | 150       | Required fields are missing or empty                                                                                                                                                                                | A three-quarters view of a red brick building with a pointed roof. The building is in the center of the image, and the side of the building facing the bottom right corner of the image is painted white. There is a gray concrete wall extending across the bottom of the image. The...                      |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`      | 20.04s    | 31.6 tok/s          | 4.6       | 500       | Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit            | - Title hint: Town centre, Halesworth, England, UK, GBR, Europe<br>- Description hint: The Cut in Halesworth, Suffolk in the UK<br>- Keyword hints: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe<br>- 10-18 unique, comma-separated keyword...             |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `unusable`      | 2.27s     | 111 tok/s           | 4.6       | 62        | Required fields are missing or empty                                                                                                                                                                                | Title: The Cut in Halesworth, England, UK, GBR, Europe<br>Description: A red brick building in Halesworth, Suffolk in the UK, with a distinctive red brick chimney and a large window. The building is surrounded by a white brick wall and has a sign on it.                                                 |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `not_evaluated` | 0.45s     | -                   | -         | -         | none                                                                                                                                                                                                                | Model loading failed: [Errno 54] Connection reset by peer                                                                                                                                                                                                                                                     |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `not_evaluated` | 12.22s    | -                   | -         | -         | none                                                                                                                                                                                                                | Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.                                                                                                                                                                         |
<!-- markdownlint-enable MD034 MD049 -->

## Avoid for This Run

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                        | Usability       | Observations                                                                                                                                                                                                        | Output preview                                                                                                                                                                                                                                                                                       |
|--------------------------------------------------------------------------------------------------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                    | `unusable`      | Required fields are missing or empty                                                                                                                                                                                | The Cut in Halesworth, Suffolk in the UK.                                                                                                                                                                                                                                                            |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                      | `unusable`      | Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible  | ◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.<br><br>The image shows a large red-brick building, which is "The Cut" in Halesworth, Suffolk,...       |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable`      | Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | Here are my reasoning steps:<br>We need to produce British-English catalogue metadata from the image and supplied context. The image shows a large brick building, likely a mill, with a sign "THE CUT" on a white-painted side. The building is red brick, multiple stories, with many...           |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable`      | Response repeats the same text; Unrecognised model control tokens remain visible; Extra text appears before the Title field; Response appears cut off at the token limit                                            | Alright, let's get to work. I've got the image and the context, and I need to generate some metadata. First, the title. The hints are a bit of a mess, but the image is clear. "The Cut, Halesworth, Suffolk" is the most accurate and concise. Done.<br><br>Next, the description. I need...        |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable`      | Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                     | A concrete 5-10-word title: Halesworth Industrial Site<br>A 1-2-sentence factual description: The Cut in Halesworth, Suffolk, UK, GBR, Europe, 2026-08-01 16:28:40 UTC+01:00<br>Keywords: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe,...        |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                               | `unusable`      | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   | <\|begin_of_box\|>Title: The Cut, Halesworth<br>Description: The image shows "The Cut" arts centre, a large red brick industrial-style building with a light blue section displaying the name and website, situated behind a brick wall with overgrown vegetation, under a blue sky with c...        |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                         | `unusable`      | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   | <\|begin_of_box\|>Title: The Cut, Halesworth, Suffolk, England<br>Description: A red brick industrial building with a light blue section displaying "THE CUT" signage, set against a blue sky with clouds, surrounded by brick walls and gravel, located in Halesworth, Suffolk, UK.<br>Keywo...     |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | `unusable`      | Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible  | ◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a title, a descriptive paragraph, and 10-18 keywords.<br><br>Starting with the title: The image shows a large red-brick building w...       |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                  | `unusable`      | Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible  | ◁think▷Okay, let's tackle this query. The user wants me to create British-English catalogue metadata based on the image and the provided context. First, I need to parse the instructions carefully. The key points are:<br><br>1. **Title**: A concrete 5-10 word title.<br>2. **Description*...    |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                   | `unusable`      | Response repeats the same text; Response appears cut off at the token limit                                                                                                                                         | Title: The Cut, Halesworth, Suffolk, England, UK, GBR, Europe<br><br>Description: A red brick building with a white sign reading "The Cut" stands prominently in the town center of Halesworth, Suffolk, England. The building, likely an industrial or historical structure, features a se...       |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                 | `unusable`      | Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                     | Industrial brick building with modern sign in town center.<br><br>Keywords: brickwork, industrial, mill, red brick, building, sign, sky, clouds, England, Europe, Suffolk, Gravel, bushes, roof, trees, Car, Clouds, England, Europe                                                                 |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                            | `unusable`      | No response text was returned; Required fields are missing or empty                                                                                                                                                 | empty output                                                                                                                                                                                                                                                                                         |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                            | `unusable`      | No response text was returned; Required fields are missing or empty                                                                                                                                                 | empty output                                                                                                                                                                                                                                                                                         |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                | `unusable`      | Required fields are missing or empty                                                                                                                                                                                | The Cut Arts Centre, Halesworth, Suffolk, UK<br><br>The Cut arts centre stands in Halesworth, Suffolk, UK, featuring a red brick building with a white side and a sign reading "The Cut" and "www.the-cut.org.uk". The structure is surrounded by a brick wall with bushes and trees, set a...       |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                  | `unusable`      | Required fields are missing or empty                                                                                                                                                                                | Title:<br>The Cut, Halesworth, Suffolk, UK, GBR, Europe<br><br>Description:<br>The Cut is a town centre arts centre located in Halesworth, Suffolk, UK. The building is a large, multi-story brick structure with a prominent sign that reads "The Cut" and a website address. The building is su... |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                  | `unusable`      | Required fields are missing or empty; Response appears cut off at the token limit                                                                                                                                   | Got it, let's tackle this step by step. First, the title. The context says "The Cut in Halesworth, Suffolk in the UK" and the image shows a red brick building with "THE CUT" sign. So the title should be concise. Maybe "The Cut, Halesworth, Suffolk, UK" but need to check if it'...             |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                          | `unusable`      | Required fields are missing or empty                                                                                                                                                                                | The Cut in Halesworth, Suffolk in the UK.                                                                                                                                                                                                                                                            |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `unusable`      | Response repeats the same text; Response appears cut off at the token limit                                                                                                                                         | Title: The Cut in Halesworth, Suffolk<br><br>Description: A historic red brick building with "The Cut" signage stands under a bright blue sky, showcasing industrial architecture in Halesworth, Suffolk, UK.<br><br>Keywords: The Cut, Halesworth, Suffolk, England, UK, Europe, Red brick build... |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | `unusable`      | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   | <\|channel>thought<br><channel\|>Title: The Cut Arts Centre in Halesworth, Suffolk<br>Description: A large multi-story red brick industrial building repurposed as an arts centre stands in Halesworth, England, under a partly cloudy sky.<br>Keywords: Halesworth, Suffolk, England, red brick...  |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | `unusable`      | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   | <\|channel>thought<br><channel\|>Title: The Cut Arts Centre Building in Halesworth, Suffolk<br>Description: A large multi-story red brick industrial building serves as The Cut arts centre, located in Halesworth under a partly cloudy sky.<br>Keywords: Halesworth, Suffolk, England, industr...  |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                  | `unusable`      | Required fields are missing or empty; Response appears cut off at the token limit                                                                                                                                   | <br>- 10-18-10-18-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-...          |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                            | `unusable`      | Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                     | The image shows a large, red brick building in Halesworth, Suffolk, UK. The building has multiple windows and a white sign that reads "THE CUT" in large letters. The building is surrounded by brick walls and some greenery. The sky is blue with some white clouds.<br><br>**Keywords:**...       |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                    | `unusable`      | No response text was returned; Required fields are missing or empty                                                                                                                                                 | empty output                                                                                                                                                                                                                                                                                         |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                 | `unusable`      | Required fields are missing or empty                                                                                                                                                                                | The Cut, Halesworth, Suffolk, England, UK, GBR, Europe, 2026-08-01 16:28:40 UTC+01:00, 52.345200°N, 1.503700°E, Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building,...             |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable`      | Required fields are missing or empty                                                                                                                                                                                | Title: The Cut in Halesworth, England, UK, GBR, Europe<br>Description: The Brickwork in Halesworth, Suffolk in the UK, featuring the Cut in the town center. The image captures the essence of the town's historic architecture, with the red brick building standing out amidst the gre...          |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)    | `unusable`      | Required fields are missing or empty                                                                                                                                                                                | Write the text in the same font as the image.                                                                                                                                                                                                                                                        |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)    | `unusable`      | Required fields are missing or empty                                                                                                                                                                                | Write the text in a plain text editor.                                                                                                                                                                                                                                                               |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)      | `unusable`      | Required fields are missing or empty                                                                                                                                                                                | A three-quarters view of a red brick building with a pointed roof. The building is in the center of the image, and the side of the building facing the bottom right corner of the image is painted white. There is a gray concrete wall extending across the bottom of the image. The...             |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable`      | Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit            | - Title hint: Town centre, Halesworth, England, UK, GBR, Europe<br>- Description hint: The Cut in Halesworth, Suffolk in the UK<br>- Keyword hints: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe<br>- 10-18 unique, comma-separated keyword...    |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                            | `unusable`      | Required fields are missing or empty                                                                                                                                                                                | Title: The Cut in Halesworth, England, UK, GBR, Europe<br>Description: A red brick building in Halesworth, Suffolk in the UK, with a distinctive red brick chimney and a large window. The building is surrounded by a white brick wall and has a sign on it.                                        |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                 | `not_evaluated` | none                                                                                                                                                                                                                | Model loading failed: [Errno 54] Connection reset by peer                                                                                                                                                                                                                                            |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                               | `not_evaluated` | none                                                                                                                                                                                                                | Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.                                                                                                                                                                |
<!-- markdownlint-enable MD034 MD049 -->

## Lowest-memory Usable Models

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability   |   Peak GB |   Gen tok |
|-------------------------------------------------------------------------------------------------------------------------|-------------|-----------|-----------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`    |       1.3 |        95 |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`    |       4.1 |       127 |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`    |       7.3 |       141 |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable`    |       8.4 |       110 |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable`    |       8.6 |       110 |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable`    |       8.6 |       110 |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `usable`    |       9.4 |        94 |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`    |       9.4 |        94 |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`    |      11   |       109 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`    |      13   |       140 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`    |      13   |       187 |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`    |      15   |       105 |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`    |      15   |        96 |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`    |      16   |       105 |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`    |      18   |        84 |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `usable`    |      18   |        89 |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`    |      18   |       132 |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable`    |      19   |       122 |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`    |      20   |       102 |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`    |      23   |       109 |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `usable`    |      25   |       114 |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `usable`    |      26   |       121 |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`    |      26   |       122 |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `usable`    |      27   |        94 |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `usable`    |      32   |       116 |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `usable`    |      35   |       114 |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `usable`    |      38   |       131 |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`    |      38   |       120 |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`    |      76   |       126 |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `usable`    |      76   |       110 |
<!-- markdownlint-enable MD034 MD049 -->

## Fastest Valid Generation

Fastest valid generation: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 511 tok/s

Average valid generation throughput: 74.2 tok/s

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability   | Gen TPS    |   Gen tok |
|-------------------------------------------------------------------------------------------------------------------------|-------------|------------|-----------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`    | 511 tok/s  |        95 |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`    | 187 tok/s  |       127 |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`    | 187 tok/s  |       141 |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`    | 120 tok/s  |       105 |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`    | 107 tok/s  |       122 |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable`    | 93.9 tok/s |       110 |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`    | 90.7 tok/s |       109 |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable`    | 89.4 tok/s |       110 |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable`    | 75.4 tok/s |       110 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`    | 67.0 tok/s |       140 |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `usable`    | 65.6 tok/s |       110 |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `usable`    | 65.0 tok/s |       114 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`    | 63.9 tok/s |       187 |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`    | 59.8 tok/s |       126 |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `usable`    | 55.8 tok/s |        94 |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`    | 53.7 tok/s |        94 |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`    | 36.9 tok/s |        96 |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`    | 34.4 tok/s |        84 |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `usable`    | 32.5 tok/s |        89 |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable`    | 32.4 tok/s |       122 |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`    | 29.5 tok/s |       132 |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`    | 27.1 tok/s |       109 |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`    | 25.9 tok/s |       102 |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `usable`    | 22.6 tok/s |       121 |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`    | 21.7 tok/s |       105 |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `usable`    | 19.8 tok/s |        94 |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `usable`    | 17.1 tok/s |       116 |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `usable`    | 14.9 tok/s |       131 |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`    | 14.2 tok/s |       120 |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `usable`    | 5.02 tok/s |       114 |
<!-- markdownlint-enable MD034 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.8`
- `mlx`: `0.32.1.dev20260802+fb5133e10`
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

## Image Metadata

- *Title:* , Town centre, Halesworth, England, UK, GBR, Europe
- *Description:* The Cut in Halesworth, Suffolk in the UK
- *Keywords:* Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork,
  Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial,
  Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk, Sunny day,
  Town, Town Centre, Tree, UK, Wall, Weeds, aged, architecture, brick, brick
  building, brick wall, building, business, city, commercial, construction,
  converted, converted warehouse, daytime, development, exterior, exterior
  view, facade, factory, heritage, historic, historic building, industrial
  architecture, industrial heritage, industrial site, landscape, masonry, old,
  old building, outdoor, parking lot, property, real estate, regeneration,
  solid, structure, summer, traditional, urban, vintage, warehouse, weathered,
  windows
- *Date:* 2026-08-01 16:28:40 UTC+01:00
- *Time:* 16:28:40
- *GPS:* 52.345200°N, 1.503700°E

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
> &#45; Capture date/time: 2026-08-01 16:28:40 UTC+01:00
> &#45; GPS: 52.345200°N, 1.503700°E
>
> &#8203;Descriptive hints:
> &#45; Title hint: Town centre, Halesworth, England, UK, GBR, Europe
> &#45; Description hint: The Cut in Halesworth, Suffolk in the UK
> &#45; Keyword hints: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork,
> Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial,
> Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk
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
- *Model load time:* 0.24s
- *Generation time:* 0.68s
- *Total time:* 1.30s
- *Input validation time:* 0.385
- *Prompt preparation time:* 0.00561
- *First-token latency:* 0.0316
- *Cleanup time:* 0.0561
- *Prompt tokens:* 380
- *Generation tokens:* 95
- *Total tokens:* 475
- *Prompt throughput (raw):* 12,008 tok/s
- *Generation throughput (raw):* 511 tok/s
- *Peak memory:* 1.3
- *Active memory:* 0.90
- *Cache memory:* 0.02
- *Model-load active memory:* 0.898
- *Post-cleanup active memory:* 0.000247
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1284
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
Title:
The Cut in Halesworth

Description:
A large brick industrial building stands prominently in the town centre of Halesworth, England, with a clear blue sky and fluffy clouds overhead.

Keywords:
Industrial building, Halesworth, England, Brickwork, Sky, Clouds, Brick, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Gravel, Halesworth, Sky, Blue sky
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
The Cut in Halesworth

Description:
A large brick industrial building stands prominently in the town centre of Halesworth, England, with a clear blue sky and fluffy clouds overhead.

Keywords:
Industrial building, Halesworth, England, Brickwork, Sky, Clouds, Brick, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Gravel, Halesworth, Sky, Blue sky
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
- *Model load time:* 0.56s
- *Generation time:* 16.02s
- *Total time:* 16.97s
- *Input validation time:* 0.383
- *Prompt preparation time:* 0.00908
- *First-token latency:* 13.9
- *Cleanup time:* 0.0919
- *Prompt tokens:* 16,581
- *Generation tokens:* 110
- *Total tokens:* 16,691
- *Prompt throughput (raw):* 1,193 tok/s
- *Generation throughput (raw):* 93.9 tok/s
- *Peak memory:* 8.6
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000558
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1305
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
Title:
The Cut, Halesworth, Suffolk

Description:
A red brick industrial building in Halesworth, Suffolk, UK, featuring a prominent white section with "THE CUT" sign, under a partly cloudy blue sky. The structure has a gravel foreground and is surrounded by greenery and a low brick wall.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Brickwork, Red Brick Building, Roof, Sky, Clouds, Gravel, Bushes, Industrial, Mill, Sign, Arts centre, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
The Cut, Halesworth, Suffolk

Description:
A red brick industrial building in Halesworth, Suffolk, UK, featuring a prominent white section with "THE CUT" sign, under a partly cloudy blue sky. The structure has a gravel foreground and is surrounded by greenery and a low brick wall.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Brickwork, Red Brick Building, Roof, Sky, Clouds, Gravel, Bushes, Industrial, Mill, Sign, Arts centre, Europe
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
- *Model load time:* 2.06s
- *Generation time:* 25.75s
- *Total time:* 28.23s
- *Input validation time:* 0.413
- *Prompt preparation time:* 0.00499
- *First-token latency:* 2.43
- *Cleanup time:* 0.0953
- *Prompt tokens:* 299
- *Generation tokens:* 114
- *Total tokens:* 413
- *Prompt throughput (raw):* 123 tok/s
- *Generation throughput (raw):* 5.02 tok/s
- *Peak memory:* 25
- *Active memory:* 21
- *Cache memory:* 3.5
- *Model-load active memory:* 21.3
- *Post-cleanup active memory:* 0.00174
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1337
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
Title: Red Brick Building with The Cut Sign in Halesworth, England

Description: A large red brick building with a distinctive roof and a prominent sign reading "The Cut" stands in Halesworth, England, under a blue sky with clouds, with a gravel area in front and a wall surrounding it. The building appears to be an industrial or commercial structure, possibly a former mill or factory.

Keywords: Red brick building, The Cut sign, Halesworth, England, industrial, commercial, blue sky, clouds, gravel, wall, distinctive roof.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Red Brick Building with The Cut Sign in Halesworth, England

Description: A large red brick building with a distinctive roof and a prominent sign reading "The Cut" stands in Halesworth, England, under a blue sky with clouds, with a gravel area in front and a wall surrounding it. The building appears to be an industrial or commercial structure, possibly a former mill or factory.

Keywords: Red brick building, The Cut sign, Halesworth, England, industrial, commercial, blue sky, clouds, gravel, wall, distinctive roof.
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
- *Model load time:* 0.74s
- *Generation time:* 2.55s
- *Total time:* 3.71s
- *Input validation time:* 0.423
- *Prompt preparation time:* 0.000936
- *First-token latency:* 0.302
- *Cleanup time:* 0.0682
- *Prompt tokens:* 1,123
- *Generation tokens:* 94
- *Total tokens:* 1,217
- *Prompt throughput (raw):* 3,716 tok/s
- *Generation throughput (raw):* 55.8 tok/s
- *Peak memory:* 9.4
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1254
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
Title: Halesworth Town Centre, Suffolk, England

Description: The Cut, a prominent red brick building in Halesworth, Suffolk, England, stands as a historical industrial landmark with a distinctive white sign on its facade.

Keywords: Halesworth, Suffolk, England, red brick building, white sign, industrial, historical, town centre, blue sky, clouds, gravel, car
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Halesworth Town Centre, Suffolk, England

Description: The Cut, a prominent red brick building in Halesworth, Suffolk, England, stands as a historical industrial landmark with a distinctive white sign on its facade.

Keywords: Halesworth, Suffolk, England, red brick building, white sign, industrial, historical, town centre, blue sky, clouds, gravel, car
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
- *Model load time:* 2.02s
- *Generation time:* 8.11s
- *Total time:* 10.60s
- *Input validation time:* 0.455
- *Prompt preparation time:* 0.00709
- *First-token latency:* 3.43
- *Cleanup time:* 0.107
- *Prompt tokens:* 2,276
- *Generation tokens:* 109
- *Total tokens:* 2,385
- *Prompt throughput (raw):* 663 tok/s
- *Generation throughput (raw):* 27.1 tok/s
- *Peak memory:* 23
- *Active memory:* 18
- *Cache memory:* 0.47
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.00233
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1233
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
Title:
The Cut Arts Centre, Halesworth

Description:
A red-brick industrial building with white-framed windows, known as The Cut Arts Centre, stands under a partly cloudy sky in Halesworth, Suffolk, UK.

Keywords:
The Cut Arts Centre, Halesworth, Suffolk, UK, red brick building, industrial architecture, white windows, partly cloudy sky, arts centre, historic mill, brickwork, gravel path, bushes, signage, blue sky, cloudy sky, European location, cultural venue
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
The Cut Arts Centre, Halesworth

Description:
A red-brick industrial building with white-framed windows, known as The Cut Arts Centre, stands under a partly cloudy sky in Halesworth, Suffolk, UK.

Keywords:
The Cut Arts Centre, Halesworth, Suffolk, UK, red brick building, industrial architecture, white windows, partly cloudy sky, arts centre, historic mill, brickwork, gravel path, bushes, signage, blue sky, cloudy sky, European location, cultural venue
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
- *Model load time:* 1.32s
- *Generation time:* 10.20s
- *Total time:* 11.94s
- *Input validation time:* 0.422
- *Prompt preparation time:* 0.000233
- *First-token latency:* 8.18
- *Cleanup time:* 0.0779
- *Prompt tokens:* 6,342
- *Generation tokens:* 110
- *Total tokens:* 6,452
- *Prompt throughput (raw):* 776 tok/s
- *Generation throughput (raw):* 75.4 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.46
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00347
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1314
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
Title: The Cut, Halesworth
Description: The image shows "The Cut" building, a large red brick industrial-style structure with a white section displaying "THE CUT" signage, located in Halesworth, Suffolk, England. The scene includes a gravel foreground, a brick wall, and greenery, under a blue sky with clouds.
Keywords: The Cut, Halesworth, Suffolk, England, UK, red brick building, industrial architecture, white signage, gravel foreground, brick wall, greenery, blue sky, clouds.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut, Halesworth
Description: The image shows "The Cut" building, a large red brick industrial-style structure with a white section displaying "THE CUT" signage, located in Halesworth, Suffolk, England. The scene includes a gravel foreground, a brick wall, and greenery, under a blue sky with clouds.
Keywords: The Cut, Halesworth, Suffolk, England, UK, red brick building, industrial architecture, white signage, gravel foreground, brick wall, greenery, blue sky, clouds.
```

</details>

</details>

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### mlx-community/Idefics3-8B-Llama3-bf16

<details>
<summary>Complete evidence: mlx-community/Idefics3-8B-Llama3-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Model load time:* 1.90s
- *Generation time:* 5.66s
- *Total time:* 7.99s
- *Input validation time:* 0.422
- *Prompt preparation time:* 0.00128
- *First-token latency:* 1.14
- *Cleanup time:* 0.0958
- *Prompt tokens:* 2,609
- *Generation tokens:* 122
- *Total tokens:* 2,731
- *Prompt throughput (raw):* 2,296 tok/s
- *Generation throughput (raw):* 32.4 tok/s
- *Peak memory:* 19
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.00405
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1270
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
 Title: The Cut in Halesworth, Suffolk, England, UK.
Description: The Cut is a historic brick building in Halesworth, Suffolk, England, UK, featuring a red brick facade with multiple windows and a white sign with the text "The Cut" on it. The building is surrounded by a brick wall and has a gravel area in front. The sky above is blue with scattered clouds.
Keywords: the cut, halesworth, suffolk, england, uk, brick building, red brick, windows, white sign, brick wall, gravel, blue sky, clouds.&lt;end_of_utterance&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Title: The Cut in Halesworth, Suffolk, England, UK.
Description: The Cut is a historic brick building in Halesworth, Suffolk, England, UK, featuring a red brick facade with multiple windows and a white sign with the text "The Cut" on it. The building is surrounded by a brick wall and has a gravel area in front. The sky above is blue with scattered clouds.
Keywords: the cut, halesworth, suffolk, england, uk, brick building, red brick, windows, white sign, brick wall, gravel, blue sky, clouds.<end_of_utterance>
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
- *Model load time:* 1.68s
- *Generation time:* 5.00s
- *Total time:* 7.10s
- *Input validation time:* 0.415
- *Prompt preparation time:* 0.00175
- *First-token latency:* 1.6
- *Cleanup time:* 0.107
- *Prompt tokens:* 2,110
- *Generation tokens:* 89
- *Total tokens:* 2,199
- *Prompt throughput (raw):* 1,322 tok/s
- *Generation throughput (raw):* 32.5 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.50
- *Model-load active memory:* 16.4
- *Post-cleanup active memory:* 0.00436
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1238
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
Title:
Red Brick Building, Halesworth, Suffolk

Description:
A red brick building in Halesworth, Suffolk, UK, with "The Cut" signage on the side, under a blue sky with scattered clouds.

Keywords:
Adobe Stock, Arts centre, Brickwork, Bushes, England, Europe, Gravel, Halesworth, Industrial, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk, Town centre
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Red Brick Building, Halesworth, Suffolk

Description:
A red brick building in Halesworth, Suffolk, UK, with "The Cut" signage on the side, under a blue sky with scattered clouds.

Keywords:
Adobe Stock, Arts centre, Brickwork, Bushes, England, Europe, Gravel, Halesworth, Industrial, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk, Town centre
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
- *Model load time:* 1.60s
- *Generation time:* 3.90s
- *Total time:* 5.93s
- *Input validation time:* 0.423
- *Prompt preparation time:* 0.0002
- *First-token latency:* 0.796
- *Cleanup time:* 0.102
- *Prompt tokens:* 2,110
- *Generation tokens:* 84
- *Total tokens:* 2,194
- *Prompt throughput (raw):* 2,651 tok/s
- *Generation throughput (raw):* 34.4 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.17
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.00467
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1238
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
Title: The Cut in Halesworth, Suffolk
Description: A red brick building with "The Cut" sign stands under a blue sky with clouds, located in Halesworth, Suffolk, UK.
Keywords: Red Brick Building, Blue Sky, Clouds, The Cut, Halesworth, Suffolk, England, Europe, Brickwork, Sign, Mill, Industrial, Bushes, Gravel, Arts Centre
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut in Halesworth, Suffolk
Description: A red brick building with "The Cut" sign stands under a blue sky with clouds, located in Halesworth, Suffolk, UK.
Keywords: Red Brick Building, Blue Sky, Clouds, The Cut, Halesworth, Suffolk, England, Europe, Brickwork, Sign, Mill, Industrial, Bushes, Gravel, Arts Centre
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
- *Model load time:* 0.54s
- *Generation time:* 1.28s
- *Total time:* 2.25s
- *Input validation time:* 0.431
- *Prompt preparation time:* 0.00352
- *First-token latency:* 0.0971
- *Cleanup time:* 0.0829
- *Prompt tokens:* 580
- *Generation tokens:* 127
- *Total tokens:* 707
- *Prompt throughput (raw):* 5,971 tok/s
- *Generation throughput (raw):* 187 tok/s
- *Peak memory:* 4.1
- *Active memory:* 3.2
- *Cache memory:* 0.02
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.00624
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1284
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
Title: The Cut, Halesworth Mill, Suffolk, England

Description: A historic red brick industrial building stands prominently in the town centre of Halesworth, Suffolk, England. The structure, known as The Cut, features a distinctive roof and a sign indicating its industrial past. The surrounding area includes a gravel road and a low brick wall, with a clear blue sky and scattered clouds overhead.

Keywords: Halesworth, Suffolk, England, The Cut, Industrial, Red Brick Building, Mill, Roof, Sign, Sky, Gravel, Brickwork, Town Centre, Blue Sky, Clouds
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut, Halesworth Mill, Suffolk, England

Description: A historic red brick industrial building stands prominently in the town centre of Halesworth, Suffolk, England. The structure, known as The Cut, features a distinctive roof and a sign indicating its industrial past. The surrounding area includes a gravel road and a low brick wall, with a clear blue sky and scattered clouds overhead.

Keywords: Halesworth, Suffolk, England, The Cut, Industrial, Red Brick Building, Mill, Roof, Sign, Sky, Gravel, Brickwork, Town Centre, Blue Sky, Clouds
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
- *Model load time:* 1.49s
- *Generation time:* 6.89s
- *Total time:* 8.82s
- *Input validation time:* 0.43
- *Prompt preparation time:* 0.00128
- *First-token latency:* 1.47
- *Cleanup time:* 0.0916
- *Prompt tokens:* 298
- *Generation tokens:* 105
- *Total tokens:* 403
- *Prompt throughput (raw):* 203 tok/s
- *Generation throughput (raw):* 21.7 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.59
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.00677
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1337
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
Title: Red Brick Building with The Cut Sign in Halesworth, England

Description: A large, red brick building with a distinctive roof and a prominent sign reading "The Cut" stands in Halesworth, England, on a sunny day with a blue sky and white clouds. The building's industrial architecture is evident in its brickwork and windows.

Keywords: Red Brick Building, The Cut, Halesworth, England, Industrial Architecture, Brickwork, Windows, Sunny Day, Blue Sky, White Clouds.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Red Brick Building with The Cut Sign in Halesworth, England

Description: A large, red brick building with a distinctive roof and a prominent sign reading "The Cut" stands in Halesworth, England, on a sunny day with a blue sky and white clouds. The building's industrial architecture is evident in its brickwork and windows.

Keywords: Red Brick Building, The Cut, Halesworth, England, Industrial Architecture, Brickwork, Windows, Sunny Day, Blue Sky, White Clouds.
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
- *Model load time:* 1.35s
- *Generation time:* 4.65s
- *Total time:* 6.42s
- *Input validation time:* 0.416
- *Prompt preparation time:* 0.00722
- *First-token latency:* 1.96
- *Cleanup time:* 0.0813
- *Prompt tokens:* 2,809
- *Generation tokens:* 140
- *Total tokens:* 2,949
- *Prompt throughput (raw):* 1,430 tok/s
- *Generation throughput (raw):* 67.0 tok/s
- *Peak memory:* 13
- *Active memory:* 8.0
- *Cache memory:* 0.56
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.00755
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3671
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
Title:
**"The Cut – Historic Brick Mill, Halesworth, Suffolk"**

Description:
A repurposed red-brick industrial mill, now housing **The Cut** arts centre, stands under a bright blue sky with scattered clouds. The three-storey structure features arched windows and weathered brickwork, surrounded by a low boundary wall and gravelled ground, captured on a clear summer afternoon in **Halesworth, Suffolk**.

Keywords:
**arts centre, brick mill, Halesworth, Suffolk, UK, industrial heritage, red brick building, arched windows, clear blue sky, scattered clouds, gravelled ground, boundary wall, historic conversion, The Cut, England, European architecture**
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
**"The Cut – Historic Brick Mill, Halesworth, Suffolk"**

Description:
A repurposed red-brick industrial mill, now housing **The Cut** arts centre, stands under a bright blue sky with scattered clouds. The three-storey structure features arched windows and weathered brickwork, surrounded by a low boundary wall and gravelled ground, captured on a clear summer afternoon in **Halesworth, Suffolk**.

Keywords:
**arts centre, brick mill, Halesworth, Suffolk, UK, industrial heritage, red brick building, arched windows, clear blue sky, scattered clouds, gravelled ground, boundary wall, historic conversion, The Cut, England, European architecture**
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
- *Model load time:* 1.38s
- *Generation time:* 5.50s
- *Total time:* 7.30s
- *Input validation time:* 0.414
- *Prompt preparation time:* 0.000228
- *First-token latency:* 1.99
- *Cleanup time:* 0.0855
- *Prompt tokens:* 2,809
- *Generation tokens:* 187
- *Total tokens:* 2,996
- *Prompt throughput (raw):* 1,411 tok/s
- *Generation throughput (raw):* 63.9 tok/s
- *Peak memory:* 13
- *Active memory:* 8.4
- *Cache memory:* 0.56
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00782
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3671
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
**Title:**
Halesworth’s Historic Brick Mill – *THE CUT* Arts Centre

**Description:**
A large, multi-storey red-brick mill building photographed on **1 August 2026** near **Halesworth, Suffolk, UK (GPS: 52.345200°N, 1.503700°E)**, now housing *THE CUT* arts centre. Its industrial brickwork, white-framed windows, and gravel-surrounded brick wall stand under a bright blue sky with scattered clouds, showcasing its repurposed heritage charm.

**Keywords:**
Halesworth, Suffolk, UK, historic mill, brick building, arts centre, *THE CUT*, industrial architecture, red brick, white windows, gravel path, brick wall, blue sky, clouds, heritage conversion, Suffolk town, UK landmark, 2026 photograph
</pre>

<details>
<summary>Exact raw output</summary>

```text
**Title:**
Halesworth’s Historic Brick Mill – *THE CUT* Arts Centre

**Description:**
A large, multi-storey red-brick mill building photographed on **1 August 2026** near **Halesworth, Suffolk, UK (GPS: 52.345200°N, 1.503700°E)**, now housing *THE CUT* arts centre. Its industrial brickwork, white-framed windows, and gravel-surrounded brick wall stand under a bright blue sky with scattered clouds, showcasing its repurposed heritage charm.

**Keywords:**
Halesworth, Suffolk, UK, historic mill, brick building, arts centre, *THE CUT*, industrial architecture, red brick, white windows, gravel path, brick wall, blue sky, clouds, heritage conversion, Suffolk town, UK landmark, 2026 photograph
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
- *Model load time:* 0.92s
- *Generation time:* 2.27s
- *Total time:* 3.61s
- *Input validation time:* 0.412
- *Prompt preparation time:* 0.00709
- *First-token latency:* 0.933
- *Cleanup time:* 0.0734
- *Prompt tokens:* 2,808
- *Generation tokens:* 141
- *Total tokens:* 2,949
- *Prompt throughput (raw):* 3,009 tok/s
- *Generation throughput (raw):* 187 tok/s
- *Peak memory:* 7.3
- *Active memory:* 2.8
- *Cache memory:* 0.38
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00808
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3670
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
Title:
**The Cut Industrial Mill in Halesworth, Suffolk**

Description:
A restored red-brick industrial mill in Halesworth, Suffolk, UK, now repurposed as *The Cut* arts centre, bathed in bright blue sky lighting with scattered clouds. The building’s two-storey brick facade, large windows, and modern signage contrast with a surrounding gravel yard and low brick wall, punctuated by greenery.

Keywords:
red-brick industrial mill, Halesworth arts centre, Suffolk UK, brickwork architecture, blue sky lighting, scattered clouds, modern signage, gravel yard, low brick wall, greenery, historic conversion, Suffolk England, The Cut location.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
**The Cut Industrial Mill in Halesworth, Suffolk**

Description:
A restored red-brick industrial mill in Halesworth, Suffolk, UK, now repurposed as *The Cut* arts centre, bathed in bright blue sky lighting with scattered clouds. The building’s two-storey brick facade, large windows, and modern signage contrast with a surrounding gravel yard and low brick wall, punctuated by greenery.

Keywords:
red-brick industrial mill, Halesworth arts centre, Suffolk UK, brickwork architecture, blue sky lighting, scattered clouds, modern signage, gravel yard, low brick wall, greenery, historic conversion, Suffolk England, The Cut location.
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
- *Model load time:* 9.98s
- *Generation time:* 59.11s
- *Total time:* 69.52s
- *Input validation time:* 0.41
- *Prompt preparation time:* 0.0139
- *First-token latency:* 54.6
- *Cleanup time:* 0.313
- *Prompt tokens:* 16,596
- *Generation tokens:* 126
- *Total tokens:* 16,722
- *Prompt throughput (raw):* 304 tok/s
- *Generation throughput (raw):* 59.8 tok/s
- *Peak memory:* 76
- *Active memory:* 70
- *Cache memory:* 0.89
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.00983
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1324
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
Title: The Cut Arts Centre, Halesworth, Suffolk
Description: A large red-brick industrial building housing The Cut arts centre in Halesworth, Suffolk, England, featuring a prominent white mural with the venue's name and website. The scene is captured under a blue sky with scattered white clouds, with a gravel foreground and brick boundary wall.
Keywords: The Cut, Halesworth, Suffolk, England, UK, Arts Centre, Red Brick, Industrial Building, Mural, Signage, Blue Sky, Clouds, Gravel, Brick Wall, Roof, Bushes, Architecture, Daylight, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut Arts Centre, Halesworth, Suffolk
Description: A large red-brick industrial building housing The Cut arts centre in Halesworth, Suffolk, England, featuring a prominent white mural with the venue's name and website. The scene is captured under a blue sky with scattered white clouds, with a gravel foreground and brick boundary wall.
Keywords: The Cut, Halesworth, Suffolk, England, UK, Arts Centre, Red Brick, Industrial Building, Mural, Signage, Blue Sky, Clouds, Gravel, Brick Wall, Roof, Bushes, Architecture, Daylight, Europe
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
- *Model load time:* 0.94s
- *Generation time:* 2.75s
- *Total time:* 4.19s
- *Input validation time:* 0.505
- *Prompt preparation time:* 0.00123
- *First-token latency:* 0.326
- *Cleanup time:* 0.1
- *Prompt tokens:* 1,123
- *Generation tokens:* 94
- *Total tokens:* 1,217
- *Prompt throughput (raw):* 3,448 tok/s
- *Generation throughput (raw):* 53.7 tok/s
- *Peak memory:* 9.4
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.0099
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1254
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
Title: Halesworth Town Centre, Suffolk, England

Description: The Cut, a prominent red brick building in Halesworth, Suffolk, England, stands as a historical industrial landmark with a distinctive white sign on its facade.

Keywords: Halesworth, Suffolk, England, red brick building, white sign, industrial, historical, town centre, blue sky, clouds, gravel, car
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Halesworth Town Centre, Suffolk, England

Description: The Cut, a prominent red brick building in Halesworth, Suffolk, England, stands as a historical industrial landmark with a distinctive white sign on its facade.

Keywords: Halesworth, Suffolk, England, red brick building, white sign, industrial, historical, town centre, blue sky, clouds, gravel, car
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
- *Model load time:* 0.84s
- *Generation time:* 22.48s
- *Total time:* 23.92s
- *Input validation time:* 0.594
- *Prompt preparation time:* 0.000266
- *First-token latency:* 20.1
- *Cleanup time:* 0.1
- *Prompt tokens:* 16,581
- *Generation tokens:* 110
- *Total tokens:* 16,691
- *Prompt throughput (raw):* 824 tok/s
- *Generation throughput (raw):* 89.4 tok/s
- *Peak memory:* 8.6
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.27
- *Post-cleanup active memory:* 0.0105
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1305
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
Title:
The Cut, Halesworth, Suffolk

Description:
A red brick industrial building in Halesworth, Suffolk, UK, featuring a prominent white section with "THE CUT" sign, under a partly cloudy blue sky. The structure has a gravel foreground and is surrounded by greenery and a low brick wall.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Brickwork, Red Brick Building, Roof, Sky, Clouds, Gravel, Bushes, Industrial, Mill, Sign, Arts centre, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
The Cut, Halesworth, Suffolk

Description:
A red brick industrial building in Halesworth, Suffolk, UK, featuring a prominent white section with "THE CUT" sign, under a partly cloudy blue sky. The structure has a gravel foreground and is surrounded by greenery and a low brick wall.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Brickwork, Red Brick Building, Roof, Sky, Clouds, Gravel, Bushes, Industrial, Mill, Sign, Arts centre, Europe
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
- *Model load time:* 2.28s
- *Generation time:* 99.02s
- *Total time:* 102.09s
- *Input validation time:* 0.754
- *Prompt preparation time:* 0.0149
- *First-token latency:* 92.6
- *Cleanup time:* 0.155
- *Prompt tokens:* 16,596
- *Generation tokens:* 121
- *Total tokens:* 16,717
- *Prompt throughput (raw):* 179 tok/s
- *Generation throughput (raw):* 22.6 tok/s
- *Peak memory:* 26
- *Active memory:* 18
- *Cache memory:* 1.9
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0113
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1324
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
Title: The Cut Arts Centre, Halesworth, Suffolk

Description: A large red brick industrial building, identified as The Cut arts centre, stands behind a weathered brick wall under a blue sky with scattered clouds. The structure features multiple rows of sash windows and a modern white extension on the right side displaying the venue's name and website.

Keywords: The Cut, Halesworth, Suffolk, Arts Centre, Red Brick, Industrial Building, Blue Sky, Clouds, Brick Wall, Windows, Gravel, Exterior, Architecture, England, United Kingdom, Europe, Summer, Daylight
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut Arts Centre, Halesworth, Suffolk

Description: A large red brick industrial building, identified as The Cut arts centre, stands behind a weathered brick wall under a blue sky with scattered clouds. The structure features multiple rows of sash windows and a modern white extension on the right side displaying the venue's name and website.

Keywords: The Cut, Halesworth, Suffolk, Arts Centre, Red Brick, Industrial Building, Blue Sky, Clouds, Brick Wall, Windows, Gravel, Exterior, Architecture, England, United Kingdom, Europe, Summer, Daylight
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
- *Model load time:* 3.27s
- *Generation time:* 111.31s
- *Total time:* 115.07s
- *Input validation time:* 0.462
- *Prompt preparation time:* 0.0178
- *First-token latency:* 101
- *Cleanup time:* 0.197
- *Prompt tokens:* 16,596
- *Generation tokens:* 131
- *Total tokens:* 16,727
- *Prompt throughput (raw):* 164 tok/s
- *Generation throughput (raw):* 14.9 tok/s
- *Peak memory:* 38
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.0118
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1324
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
Title:
The Cut Arts Centre, Halesworth, Suffolk

Description:
A large, multi-storey red brick industrial building, identified by a white sign as "The Cut," stands behind a weathered brick and concrete wall under a blue sky with scattered clouds. The structure features rows of sash windows and a slate roof, with green vegetation growing along the base of the foreground wall.

Keywords:
Halesworth, Suffolk, The Cut, Arts Centre, Red Brick, Industrial Building, Blue Sky, Clouds, Brick Wall, Windows, Exterior, England, UK, Architecture, Gravel, Bushes, Heritage, Summer
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
The Cut Arts Centre, Halesworth, Suffolk

Description:
A large, multi-storey red brick industrial building, identified by a white sign as "The Cut," stands behind a weathered brick and concrete wall under a blue sky with scattered clouds. The structure features rows of sash windows and a slate roof, with green vegetation growing along the base of the foreground wall.

Keywords:
Halesworth, Suffolk, The Cut, Arts Centre, Red Brick, Industrial Building, Blue Sky, Clouds, Brick Wall, Windows, Exterior, England, UK, Architecture, Gravel, Bushes, Heritage, Summer
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
- *Model load time:* 2.82s
- *Generation time:* 73.79s
- *Total time:* 77.30s
- *Input validation time:* 0.678
- *Prompt preparation time:* 0.000316
- *First-token latency:* 71.6
- *Cleanup time:* 0.114
- *Prompt tokens:* 16,596
- *Generation tokens:* 122
- *Total tokens:* 16,718
- *Prompt throughput (raw):* 232 tok/s
- *Generation throughput (raw):* 107 tok/s
- *Peak memory:* 26
- *Active memory:* 21
- *Cache memory:* 0.89
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.0124
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1324
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
Title: The Cut Arts Centre Halesworth Red Brick Building

Description: A large red brick industrial building converted into The Cut arts centre, featuring a white painted gable end with signage, set against a blue sky with scattered clouds. The structure is situated behind a weathered brick and concrete wall with gravel foreground, located in Halesworth, Suffolk.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Red Brick, Arts Centre, Industrial Building, Blue Sky, Clouds, Brickwork, Wall, Gravel, Roof, Windows, Signage, Architecture, Daylight
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut Arts Centre Halesworth Red Brick Building

Description: A large red brick industrial building converted into The Cut arts centre, featuring a white painted gable end with signage, set against a blue sky with scattered clouds. The structure is situated behind a weathered brick and concrete wall with gravel foreground, located in Halesworth, Suffolk.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Red Brick, Arts Centre, Industrial Building, Blue Sky, Clouds, Brickwork, Wall, Gravel, Roof, Windows, Signage, Architecture, Daylight
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
- *Model load time:* 3.61s
- *Generation time:* 73.58s
- *Total time:* 77.61s
- *Input validation time:* 0.416
- *Prompt preparation time:* 0.000252
- *First-token latency:* 70.9
- *Cleanup time:* 0.187
- *Prompt tokens:* 16,596
- *Generation tokens:* 114
- *Total tokens:* 16,710
- *Prompt throughput (raw):* 234 tok/s
- *Generation throughput (raw):* 65.0 tok/s
- *Peak memory:* 35
- *Active memory:* 30
- *Cache memory:* 0.88
- *Model-load active memory:* 29.1
- *Post-cleanup active memory:* 0.0129
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1324
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
Title: The Cut Arts Centre Halesworth Red Brick Building

Description: A large red brick industrial building converted into an arts centre, featuring a prominent white sign reading "THE CUT" on its side. The structure stands behind a brick wall under a blue sky with scattered clouds, with gravel ground in the foreground.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Arts Centre, Red Brick, Industrial Building, Blue Sky, Clouds, Brick Wall, Gravel, Windows, Roof, Architecture, Exterior, Daylight, Community Venue
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut Arts Centre Halesworth Red Brick Building

Description: A large red brick industrial building converted into an arts centre, featuring a prominent white sign reading "THE CUT" on its side. The structure stands behind a brick wall under a blue sky with scattered clouds, with gravel ground in the foreground.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Arts Centre, Red Brick, Industrial Building, Blue Sky, Clouds, Brick Wall, Gravel, Windows, Roof, Architecture, Exterior, Daylight, Community Venue
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
- *Model load time:* 13.07s
- *Generation time:* 81.04s
- *Total time:* 94.78s
- *Input validation time:* 0.663
- *Prompt preparation time:* 0.000258
- *First-token latency:* 78.4
- *Cleanup time:* 0.254
- *Prompt tokens:* 16,596
- *Generation tokens:* 110
- *Total tokens:* 16,706
- *Prompt throughput (raw):* 212 tok/s
- *Generation throughput (raw):* 65.6 tok/s
- *Peak memory:* 76
- *Active memory:* 70
- *Cache memory:* 0.89
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.0134
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1324
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
Title: The Cut Arts Centre Halesworth Red Brick Building

Description: A large red brick industrial building converted into an arts centre, featuring a prominent white sign reading "THE CUT" against a blue sky with scattered clouds. The structure stands behind a brick wall with gravel foreground and green vegetation.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Arts Centre, Red Brick Building, Industrial, Mill, Blue Sky, Clouds, Brickwork, Sign, Gravel, Bushes, Europe, Architecture, Daylight
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut Arts Centre Halesworth Red Brick Building

Description: A large red brick industrial building converted into an arts centre, featuring a prominent white sign reading "THE CUT" against a blue sky with scattered clouds. The structure stands behind a brick wall with gravel foreground and green vegetation.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Arts Centre, Red Brick Building, Industrial, Mill, Blue Sky, Clouds, Brickwork, Sign, Gravel, Bushes, Europe, Architecture, Daylight
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
- *Model load time:* 1.41s
- *Generation time:* 71.44s
- *Total time:* 73.28s
- *Input validation time:* 0.427
- *Prompt preparation time:* 0.000258
- *First-token latency:* 69.3
- *Cleanup time:* 0.0864
- *Prompt tokens:* 16,596
- *Generation tokens:* 109
- *Total tokens:* 16,705
- *Prompt throughput (raw):* 240 tok/s
- *Generation throughput (raw):* 90.7 tok/s
- *Peak memory:* 11
- *Active memory:* 6.7
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.0139
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1324
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
Title: The Cut Arts Centre, Halesworth, Suffolk, UK

Description: A red brick industrial building with a white sign reading "THE CUT" and website URL, set against a blue sky with scattered clouds, located in Halesworth, Suffolk, England.

Keywords: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut Arts Centre, Halesworth, Suffolk, UK

Description: A red brick industrial building with a white sign reading "THE CUT" and website URL, set against a blue sky with scattered clouds, located in Halesworth, Suffolk, England.

Keywords: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk
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
- *Model load time:* 3.12s
- *Generation time:* 113.54s
- *Total time:* 117.12s
- *Input validation time:* 0.423
- *Prompt preparation time:* 0.0149
- *First-token latency:* 104
- *Cleanup time:* 0.254
- *Prompt tokens:* 16,596
- *Generation tokens:* 120
- *Total tokens:* 16,716
- *Prompt throughput (raw):* 159 tok/s
- *Generation throughput (raw):* 14.2 tok/s
- *Peak memory:* 38
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.0144
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1324
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
Title:
The Cut Arts Centre in Halesworth, Suffolk

Description:
A large, historic red brick industrial building, identified as The Cut arts centre, stands behind a weathered brick wall and concrete barrier. The scene is set under a bright blue sky with scattered white clouds, highlighting the building's multiple windows and the gravel foreground.

Keywords:
Halesworth, Suffolk, England, The Cut, Arts centre, Red brick, Industrial building, Blue sky, Clouds, Brick wall, Gravel, Bushes, Windows, Architecture, UK, Europe, Heritage, Exterior
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
The Cut Arts Centre in Halesworth, Suffolk

Description:
A large, historic red brick industrial building, identified as The Cut arts centre, stands behind a weathered brick wall and concrete barrier. The scene is set under a bright blue sky with scattered white clouds, highlighting the building's multiple windows and the gravel foreground.

Keywords:
Halesworth, Suffolk, England, The Cut, Arts centre, Red brick, Industrial building, Blue sky, Clouds, Brick wall, Gravel, Bushes, Windows, Architecture, UK, Europe, Heritage, Exterior
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
- *Model load time:* 2.40s
- *Generation time:* 6.17s
- *Total time:* 9.00s
- *Input validation time:* 0.422
- *Prompt preparation time:* 0.0036
- *First-token latency:* 1.07
- *Cleanup time:* 0.125
- *Prompt tokens:* 591
- *Generation tokens:* 132
- *Total tokens:* 723
- *Prompt throughput (raw):* 554 tok/s
- *Generation throughput (raw):* 29.5 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.44
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0154
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1288
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
Title: The Cut, Halesworth: Red Brick Arts Centre, Suffolk
Description: Captured on 1st August 2026, this image shows The Cut, a large red-brick building in Halesworth, Suffolk, under a clear blue sky; a sign displays the venue’s website address. A gravelled area and brick wall feature in the foreground, with a car visible in the distance.
Keywords: Halesworth, Suffolk, England, The Cut, Arts centre, Red brick building, Architecture, Exterior, Gravel, Sign, Blue sky, Clouds, Car, Mill, Brickwork, Daylight, Urban, Building
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut, Halesworth: Red Brick Arts Centre, Suffolk
Description: Captured on 1st August 2026, this image shows The Cut, a large red-brick building in Halesworth, Suffolk, under a clear blue sky; a sign displays the venue’s website address. A gravelled area and brick wall feature in the foreground, with a car visible in the distance.
Keywords: Halesworth, Suffolk, England, The Cut, Arts centre, Red brick building, Architecture, Exterior, Gravel, Sign, Blue sky, Clouds, Car, Mill, Brickwork, Daylight, Urban, Building
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
- *Model load time:* 3.16s
- *Generation time:* 8.67s
- *Total time:* 12.25s
- *Input validation time:* 0.42
- *Prompt preparation time:* 0.00022
- *First-token latency:* 1.27
- *Cleanup time:* 0.168
- *Prompt tokens:* 591
- *Generation tokens:* 116
- *Total tokens:* 707
- *Prompt throughput (raw):* 464 tok/s
- *Generation throughput (raw):* 17.1 tok/s
- *Peak memory:* 32
- *Active memory:* 30
- *Cache memory:* 0.44
- *Model-load active memory:* 29.7
- *Post-cleanup active memory:* 0.0159
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1288
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
Title: The Cut, Halesworth: Red Brick Arts Centre, Suffolk
Description: Captured on 1st August 2026, The Cut arts centre stands in Halesworth, Suffolk, bathed in bright sunlight. The large red brick building features a distinctive gabled roof and a prominent sign displaying its website address.
Keywords: Halesworth, Suffolk, The Cut, Arts centre, Red brick building, Architecture, Gabled roof, Sign, Website, Exterior, Daylight, Blue sky, Clouds, Gravel, Brickwork, England, UK, Building
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut, Halesworth: Red Brick Arts Centre, Suffolk
Description: Captured on 1st August 2026, The Cut arts centre stands in Halesworth, Suffolk, bathed in bright sunlight. The large red brick building features a distinctive gabled roof and a prominent sign displaying its website address.
Keywords: Halesworth, Suffolk, The Cut, Arts centre, Red brick building, Architecture, Gabled roof, Sign, Website, Exterior, Daylight, Blue sky, Clouds, Gravel, Brickwork, England, UK, Building
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
- *Model load time:* 2.41s
- *Generation time:* 1.96s
- *Total time:* 4.82s
- *Input validation time:* 0.415
- *Prompt preparation time:* 0.0213
- *First-token latency:* 0.467
- *Cleanup time:* 0.126
- *Prompt tokens:* 609
- *Generation tokens:* 105
- *Total tokens:* 714
- *Prompt throughput (raw):* 1,305 tok/s
- *Generation throughput (raw):* 120 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.21
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0175
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1287
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
Title: Red brick building The Cut in Halesworth, Suffolk

Description: A large, historic red brick industrial building known as The Cut stands under a blue sky with white clouds in Halesworth, Suffolk. The multi-storey structure features various window styles and a large white sign on its side.

Keywords: Halesworth, Suffolk, England, UK, red brick building, The Cut, industrial architecture, historic building, blue sky, clouds, brickwork, Europe, daytime, architecture, building, outdoors
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Red brick building The Cut in Halesworth, Suffolk

Description: A large, historic red brick industrial building known as The Cut stands under a blue sky with white clouds in Halesworth, Suffolk. The multi-storey structure features various window styles and a large white sign on its side.

Keywords: Halesworth, Suffolk, England, UK, red brick building, The Cut, industrial architecture, historic building, blue sky, clouds, brickwork, Europe, daytime, architecture, building, outdoors
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
- *Model load time:* 2.69s
- *Generation time:* 5.73s
- *Total time:* 8.86s
- *Input validation time:* 0.436
- *Prompt preparation time:* 0.000334
- *First-token latency:* 1.14
- *Cleanup time:* 0.136
- *Prompt tokens:* 609
- *Generation tokens:* 102
- *Total tokens:* 711
- *Prompt throughput (raw):* 536 tok/s
- *Generation throughput (raw):* 25.9 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.79
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0186
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1287
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
Title: The Cut Arts Centre in Halesworth, Suffolk
Description: A large, multi-storey red brick industrial mill building, now The Cut arts centre, under a blue sky with scattered clouds in Halesworth, Suffolk. The structure features numerous small windows and a large white sign on its side.
Keywords: The Cut, Halesworth, Suffolk, England, arts centre, red brick, industrial, mill, architecture, brickwork, sign, blue sky, clouds, Europe, UK
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut Arts Centre in Halesworth, Suffolk
Description: A large, multi-storey red brick industrial mill building, now The Cut arts centre, under a blue sky with scattered clouds in Halesworth, Suffolk. The structure features numerous small windows and a large white sign on its side.
Keywords: The Cut, Halesworth, Suffolk, England, arts centre, red brick, industrial, mill, architecture, brickwork, sign, blue sky, clouds, Europe, UK
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
- *Model load time:* 1.84s
- *Generation time:* 5.35s
- *Total time:* 7.69s
- *Input validation time:* 0.489
- *Prompt preparation time:* 0.0042
- *First-token latency:* 2.04
- *Cleanup time:* 0.172
- *Prompt tokens:* 2,916
- *Generation tokens:* 96
- *Total tokens:* 3,012
- *Prompt throughput (raw):* 1,432 tok/s
- *Generation throughput (raw):* 36.9 tok/s
- *Peak memory:* 15
- *Active memory:* 14
- *Cache memory:* 0.55
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0236
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1233
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
Title:
Red Brick Mill Building in Halesworth, Suffolk

Description:
A red brick mill building in Halesworth, Suffolk, stands prominently under a blue sky with scattered clouds, featuring the sign "The Cut" on its facade.

Keywords:
Red Brick Building, Halesworth, Suffolk, Mill, Industrial, Brickwork, Blue Sky, Clouds, England, Europe, Gravel, Sign, The Cut, Bushes, Roof, Windows, Brick Wall
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Red Brick Mill Building in Halesworth, Suffolk

Description:
A red brick mill building in Halesworth, Suffolk, stands prominently under a blue sky with scattered clouds, featuring the sign "The Cut" on its facade.

Keywords:
Red Brick Building, Halesworth, Suffolk, Mill, Industrial, Brickwork, Blue Sky, Clouds, England, Europe, Gravel, Sign, The Cut, Bushes, Roof, Windows, Brick Wall
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
- *Model load time:* 2.69s
- *Generation time:* 7.37s
- *Total time:* 10.82s
- *Input validation time:* 0.754
- *Prompt preparation time:* 0.00024
- *First-token latency:* 1.99
- *Cleanup time:* 0.121
- *Prompt tokens:* 2,916
- *Generation tokens:* 94
- *Total tokens:* 3,010
- *Prompt throughput (raw):* 1,468 tok/s
- *Generation throughput (raw):* 19.8 tok/s
- *Peak memory:* 27
- *Active memory:* 25
- *Cache memory:* 0.54
- *Model-load active memory:* 25.4
- *Post-cleanup active memory:* 0.0239
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1233
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
Title:
Red Brick Mill Building in Halesworth, Suffolk

Description:
A red brick mill building in Halesworth, Suffolk, stands prominently under a blue sky with scattered clouds, featuring a sign that reads "The Cut."

Keywords:
Red Brick Building, Halesworth, Suffolk, Mill, Industrial, Brickwork, Blue Sky, Clouds, England, Europe, Gravel, Sign, The Cut, Bushes, Roof, Windows, Brick Wall
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Red Brick Mill Building in Halesworth, Suffolk

Description:
A red brick mill building in Halesworth, Suffolk, stands prominently under a blue sky with scattered clouds, featuring a sign that reads "The Cut."

Keywords:
Red Brick Building, Halesworth, Suffolk, Mill, Industrial, Brickwork, Blue Sky, Clouds, England, Europe, Gravel, Sign, The Cut, Bushes, Roof, Windows, Brick Wall
```

</details>

</details>

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### HuggingFaceTB/SmolVLM-Instruct

<details>
<summary>Complete evidence: HuggingFaceTB/SmolVLM-Instruct</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty
- *Model load time:* 12.11s
- *Generation time:* 1.23s
- *Total time:* 13.73s
- *Input validation time:* 0.389
- *Prompt preparation time:* 0.00198
- *First-token latency:* 0.395
- *Cleanup time:* 0.0636
- *Prompt tokens:* 1,522
- *Generation tokens:* 13
- *Total tokens:* 1,535
- *Prompt throughput (raw):* 3,850 tok/s
- *Generation throughput (raw):* 131 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.35
- *Model-load active memory:* 4.49
- *Post-cleanup active memory:* 0.000116
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1265
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
 The Cut in Halesworth, Suffolk in the UK.
</pre>

<details>
<summary>Exact raw output</summary>

```text
 The Cut in Halesworth, Suffolk in the UK.
```

</details>

</details>

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### jqlive/Kimi-VL-A3B-Thinking-2506-6bit

<details>
<summary>Complete evidence: jqlive/Kimi-VL-A3B-Thinking-2506-6bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty; Extra text appears
  before the Title field; Response appears cut off at the token limit;
  Internal reasoning block appears incomplete; Internal reasoning text remains
  visible
- *Model load time:* 1.75s
- *Generation time:* 8.45s
- *Total time:* 10.64s
- *Input validation time:* 0.429
- *Prompt preparation time:* 0.00183
- *First-token latency:* 0.866
- *Cleanup time:* 0.0885
- *Prompt tokens:* 1,308
- *Generation tokens:* 500
- *Total tokens:* 1,808
- *Prompt throughput (raw):* 1,510 tok/s
- *Generation throughput (raw):* 72.6 tok/s
- *Peak memory:* 18
- *Active memory:* 14
- *Cache memory:* 1.4
- *Model-load active memory:* 13.8
- *Post-cleanup active memory:* 0.00121
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1420
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
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.

The image shows a large red-brick building, which is "The Cut" in Halesworth, Suffolk, UK. The context mentions the capture date/time and GPS, but we treat those as authoritative but not visible. The descriptive hints include elements like brickwork, blue sky, bushes, gravel, etc.

First, the title: need a 5-10 word title. Let's see, the building is a red-brick industrial building, so maybe "Red-brick industrial building in Halesworth, Suffolk" but need to make it concise. Wait, the hint says "The Cut in Halesworth, Suffolk in the UK" so maybe "The Cut, Halesworth, Suffolk: Red-brick industrial building" but check word count. Let's count: "The Cut, Halesworth, Suffolk: Red-brick industrial building" is 10 words. Good.

Description: 1-2 sentences. The main subject is the red-brick building, setting is town centre, Halesworth, England. Visible details: blue sky, brickwork, bushes, gravel. So "A large red-brick industrial building with white facade, situated in Halesworth's town centre, featuring brickwork, blue sky, bushes, and gravel under a partly cloudy sky." Wait, the image has a blue sky with clouds, so "blue sky" and "clouds" are visible. Also, the building has a white facade with "THE CUT" sign. So maybe: "A large red-brick industrial building with a white facade displaying 'THE CUT' signage, located in Halesworth's town centre, England, featuring brickwork, blue sky, bushes, and gravel under a partly cloudy sky."

Keywords: 10-18 unique keywords. Let's list from the hints and image: Adobe Stock, Any Vision, Arts centre (maybe not visible, but hint says Arts centre, so include), Blue sky, Brickwork, Bushes, Car (no car visible, so omit), Clouds, England, Europe, Gravel, Halesworth, Industrial, Mill (maybe, since it's an old industrial building), Red Brick Building, Roof (the roof is visible), Sign (THE CUT sign), Sky, Suffolk. Wait, the
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.

The image shows a large red-brick building, which is "The Cut" in Halesworth, Suffolk, UK. The context mentions the capture date/time and GPS, but we treat those as authoritative but not visible. The descriptive hints include elements like brickwork, blue sky, bushes, gravel, etc.

First, the title: need a 5-10 word title. Let's see, the building is a red-brick industrial building, so maybe "Red-brick industrial building in Halesworth, Suffolk" but need to make it concise. Wait, the hint says "The Cut in Halesworth, Suffolk in the UK" so maybe "The Cut, Halesworth, Suffolk: Red-brick industrial building" but check word count. Let's count: "The Cut, Halesworth, Suffolk: Red-brick industrial building" is 10 words. Good.

Description: 1-2 sentences. The main subject is the red-brick building, setting is town centre, Halesworth, England. Visible details: blue sky, brickwork, bushes, gravel. So "A large red-brick industrial building with white facade, situated in Halesworth's town centre, featuring brickwork, blue sky, bushes, and gravel under a partly cloudy sky." Wait, the image has a blue sky with clouds, so "blue sky" and "clouds" are visible. Also, the building has a white facade with "THE CUT" sign. So maybe: "A large red-brick industrial building with a white facade displaying 'THE CUT' signage, located in Halesworth's town centre, England, featuring brickwork, blue sky, bushes, and gravel under a partly cloudy sky."

Keywords: 10-18 unique keywords. Let's list from the hints and image: Adobe Stock, Any Vision, Arts centre (maybe not visible, but hint says Arts centre, so include), Blue sky, Brickwork, Bushes, Car (no car visible, so omit), Clouds, England, Europe, Gravel, Halesworth, Industrial, Mill (maybe, since it's an old industrial building), Red Brick Building, Roof (the roof is visible), Sign (THE CUT sign), Sky, Suffolk. Wait, the
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
- *Observations:* Required fields are missing or empty; Response repeats the
  task instructions instead of only returning the requested fields; Extra text
  appears before the Title field; Response appears cut off at the token limit
- *Model load time:* 1.44s
- *Generation time:* 14.67s
- *Total time:* 16.54s
- *Input validation time:* 0.416
- *Prompt preparation time:* 0.00678
- *First-token latency:* 2.01
- *Cleanup time:* 0.104
- *Prompt tokens:* 3,007
- *Generation tokens:* 500
- *Total tokens:* 3,507
- *Prompt throughput (raw):* 1,495 tok/s
- *Generation throughput (raw):* 41.4 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.72
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.00207
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1607
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
We need to produce British-English catalogue metadata from the image and supplied context. The image shows a large brick building, likely a mill, with a sign "THE CUT" on a white-painted side. The building is red brick, multiple stories, with many windows, some small, some larger. There's a brick wall in front, some vegetation, a gravel ground. The sky is partly cloudy with blue sky. There's a car partially visible? In the image, I see a car partially visible on the left side? Actually, there is a car partially visible near the left side of the building? Let's examine the image: The building is a large red brick structure, with a white side that has "THE CUT" sign. The building appears to be an industrial or mill building, possibly a converted arts centre. The surroundings include a brick wall, some vegetation, a gravel ground, and a partly cloudy blue sky. There's a car visible near the left side? The image shows a car partially visible behind the building? Actually, I see a car on the left side of the image, near the building's left side, maybe parked. The building is in Halesworth, Suffolk, England. The sign "THE CUT" suggests a location or perhaps a business. The building appears to be a former mill, now possibly an arts centre. The building has a pitched roof, with some chimneys? The building is large, with many windows, some with white frames. The building is surrounded by a brick wall. The ground is gravel. There's a small patch of grass and bushes. The sky is partly cloudy with blue sky.

We need to produce:

- Title: 5-10 words, concrete. Something like "The Cut – Halesworth Red Brick Mill". That's 6 words: "The Cut – Halesworth Red Brick Mill". Or "Red Brick Mill, The Cut, Halesworth". That's 6 words. Or "The Cut, Halesworth, Red Brick Mill". That's 6 words. Or "Halesworth Mill – The Cut". That's 5 words. Let's choose "The Cut – Halesworth Red Brick Mill". That's 6 words (including hyphen maybe counts as part of word). Title: "The Cut – Halesworth Red Brick Mill". That is okay.

- Description: 1-2
</pre>

<details>
<summary>Exact raw output</summary>

```text
Here are my reasoning steps:
We need to produce British-English catalogue metadata from the image and supplied context. The image shows a large brick building, likely a mill, with a sign "THE CUT" on a white-painted side. The building is red brick, multiple stories, with many windows, some small, some larger. There's a brick wall in front, some vegetation, a gravel ground. The sky is partly cloudy with blue sky. There's a car partially visible? In the image, I see a car partially visible on the left side? Actually, there is a car partially visible near the left side of the building? Let's examine the image: The building is a large red brick structure, with a white side that has "THE CUT" sign. The building appears to be an industrial or mill building, possibly a converted arts centre. The surroundings include a brick wall, some vegetation, a gravel ground, and a partly cloudy blue sky. There's a car visible near the left side? The image shows a car partially visible behind the building? Actually, I see a car on the left side of the image, near the building's left side, maybe parked. The building is in Halesworth, Suffolk, England. The sign "THE CUT" suggests a location or perhaps a business. The building appears to be a former mill, now possibly an arts centre. The building has a pitched roof, with some chimneys? The building is large, with many windows, some with white frames. The building is surrounded by a brick wall. The ground is gravel. There's a small patch of grass and bushes. The sky is partly cloudy with blue sky.

We need to produce:

- Title: 5-10 words, concrete. Something like "The Cut – Halesworth Red Brick Mill". That's 6 words: "The Cut – Halesworth Red Brick Mill". Or "Red Brick Mill, The Cut, Halesworth". That's 6 words. Or "The Cut, Halesworth, Red Brick Mill". That's 6 words. Or "Halesworth Mill – The Cut". That's 5 words. Let's choose "The Cut – Halesworth Red Brick Mill". That's 6 words (including hyphen maybe counts as part of word). Title: "The Cut – Halesworth Red Brick Mill". That is okay.

- Description: 1-2
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
- *Observations:* Response repeats the same text; Unrecognised model control
  tokens remain visible; Extra text appears before the Title field; Response
  appears cut off at the token limit
- *Model load time:* 10.61s
- *Generation time:* 16.65s
- *Total time:* 27.72s
- *Input validation time:* 0.453
- *Prompt preparation time:* 0.00801
- *First-token latency:* 6.28
- *Cleanup time:* 0.342
- *Prompt tokens:* 1,641
- *Generation tokens:* 500
- *Total tokens:* 2,141
- *Prompt throughput (raw):* 261 tok/s
- *Generation throughput (raw):* 52.4 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.14
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00254
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1423
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
Alright, let's get to work. I've got the image and the context, and I need to generate some metadata. First, the title. The hints are a bit of a mess, but the image is clear. "The Cut, Halesworth, Suffolk" is the most accurate and concise. Done.

Next, the description. I need to weave in the context – the date, time, and location – with the visual details. The image shows a red brick building, likely an old industrial site, with a sign. The hints mention "The Cut," "Halesworth," and "Suffolk," so that's where I'll start. I'll add the date and time, and describe the building's appearance and the sign. I'll also mention the sky and the general setting. I'll make sure to include the architectural style and the sign.

Finally, the keywords. I'll pull from the hints, but I need to be selective. I'll focus on the location, the building itself, the sign, the sky, and the general setting. I'll avoid anything that's not clearly visible or relevant. I'll make sure to include the architectural style, the sign, the sky, and the general setting. I'll also add the date and time.

Okay, let's put it all together.

**Title:** The Cut, Halesworth, Suffolk

**Description:** A photograph of The Cut, an industrial red brick building in Halesworth, Suffolk, UK, taken on August 1st, 2026. The building features a sign and is set against a blue sky with fluffy clouds.

**Keywords:** The Cut, Halesworth, Suffolk, UK, England, Red Brick Building, Industrial, Mill, Sign, Blue Sky, Clouds, Gravel, Bushes, Adobe Stock, Arts centre, Location, Mill, Roof, Sky, Suffolk, UK, Europe, Gravel, Bushes, Car, Clouds, England, Europe, Brickwork, Sign, Sky, Industrial, Mill, Red Brick Building, Roof, Location.
&lt;/think&gt;

Title: The Cut, Halesworth, Suffolk

Description: A photograph of The Cut, an industrial red brick building in Halesworth, Suff
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get to work. I've got the image and the context, and I need to generate some metadata. First, the title. The hints are a bit of a mess, but the image is clear. "The Cut, Halesworth, Suffolk" is the most accurate and concise. Done.

Next, the description. I need to weave in the context – the date, time, and location – with the visual details. The image shows a red brick building, likely an old industrial site, with a sign. The hints mention "The Cut," "Halesworth," and "Suffolk," so that's where I'll start. I'll add the date and time, and describe the building's appearance and the sign. I'll also mention the sky and the general setting. I'll make sure to include the architectural style and the sign.

Finally, the keywords. I'll pull from the hints, but I need to be selective. I'll focus on the location, the building itself, the sign, the sky, and the general setting. I'll avoid anything that's not clearly visible or relevant. I'll make sure to include the architectural style, the sign, the sky, and the general setting. I'll also add the date and time.

Okay, let's put it all together.

**Title:** The Cut, Halesworth, Suffolk

**Description:** A photograph of The Cut, an industrial red brick building in Halesworth, Suffolk, UK, taken on August 1st, 2026. The building features a sign and is set against a blue sky with fluffy clouds.

**Keywords:** The Cut, Halesworth, Suffolk, UK, England, Red Brick Building, Industrial, Mill, Sign, Blue Sky, Clouds, Gravel, Bushes, Adobe Stock, Arts centre, Location, Mill, Roof, Sky, Suffolk, UK, Europe, Gravel, Bushes, Car, Clouds, England, Europe, Brickwork, Sign, Sky, Industrial, Mill, Red Brick Building, Roof, Location.
</think>

Title: The Cut, Halesworth, Suffolk

Description: A photograph of The Cut, an industrial red brick building in Halesworth, Suff
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
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty; Extra text appears
  before the Title field
- *Model load time:* 0.68s
- *Generation time:* 1.87s
- *Total time:* 3.25s
- *Input validation time:* 0.7
- *Prompt preparation time:* 0.00175
- *First-token latency:* 0.103
- *Cleanup time:* 0.0936
- *Prompt tokens:* 331
- *Generation tokens:* 124
- *Total tokens:* 455
- *Prompt throughput (raw):* 3,227 tok/s
- *Generation throughput (raw):* 331 tok/s
- *Peak memory:* 2.1
- *Active memory:* 1.2
- *Cache memory:* 0.02
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.00285
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1328
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
A concrete 5-10-word title: Halesworth Industrial Site
A 1-2-sentence factual description: The Cut in Halesworth, Suffolk, UK, GBR, Europe, 2026-08-01 16:28:40 UTC+01:00
Keywords: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk
</pre>

<details>
<summary>Exact raw output</summary>

```text
A concrete 5-10-word title: Halesworth Industrial Site
A 1-2-sentence factual description: The Cut in Halesworth, Suffolk, UK, GBR, Europe, 2026-08-01 16:28:40 UTC+01:00
Keywords: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk
```

</details>

</details>

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### mlx-community/GLM-4.6V-Flash-6bit

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-6bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible; Required
  fields are missing or empty; Extra text appears before the Title field
- *Model load time:* 1.55s
- *Generation time:* 12.77s
- *Total time:* 14.76s
- *Input validation time:* 0.43
- *Prompt preparation time:* 0.00711
- *First-token latency:* 10.1
- *Cleanup time:* 0.0828
- *Prompt tokens:* 6,342
- *Generation tokens:* 106
- *Total tokens:* 6,448
- *Prompt throughput (raw):* 626 tok/s
- *Generation throughput (raw):* 52.5 tok/s
- *Peak memory:* 11
- *Active memory:* 9.4
- *Cache memory:* 0.45
- *Model-load active memory:* 9.43
- *Post-cleanup active memory:* 0.00316
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1314
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
&lt;|begin_of_box|&gt;Title: The Cut, Halesworth
Description: The image shows "The Cut" arts centre, a large red brick industrial-style building with a light blue section displaying the name and website, situated behind a brick wall with overgrown vegetation, under a blue sky with clouds, in Halesworth, Suffolk, England.
Keywords: The Cut, Halesworth, arts centre, red brick building, light blue section, brick wall, overgrown vegetation, blue sky, clouds, Suffolk, England, UK.&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: The Cut, Halesworth
Description: The image shows "The Cut" arts centre, a large red brick industrial-style building with a light blue section displaying the name and website, situated behind a brick wall with overgrown vegetation, under a blue sky with clouds, in Halesworth, Suffolk, England.
Keywords: The Cut, Halesworth, arts centre, red brick building, light blue section, brick wall, overgrown vegetation, blue sky, clouds, Suffolk, England, UK.<|end_of_box|>
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
- *Observations:* Unrecognised model control tokens remain visible; Required
  fields are missing or empty; Extra text appears before the Title field
- *Model load time:* 9.31s
- *Generation time:* 26.37s
- *Total time:* 36.12s
- *Input validation time:* 0.42
- *Prompt preparation time:* 0.00691
- *First-token latency:* 22.8
- *Cleanup time:* 0.246
- *Prompt tokens:* 6,342
- *Generation tokens:* 104
- *Total tokens:* 6,446
- *Prompt throughput (raw):* 278 tok/s
- *Generation throughput (raw):* 36.5 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00379
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1314
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
&lt;|begin_of_box|&gt;Title: The Cut, Halesworth, Suffolk, England
Description: A red brick industrial building with a light blue section displaying "THE CUT" signage, set against a blue sky with clouds, surrounded by brick walls and gravel, located in Halesworth, Suffolk, UK.
Keywords: The Cut, Halesworth, Suffolk, England, UK, Europe, Arts centre, Red Brick Building, Brickwork, Blue sky, Clouds, Gravel, Industrial, Mill, Sign, Bushes&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: The Cut, Halesworth, Suffolk, England
Description: A red brick industrial building with a light blue section displaying "THE CUT" signage, set against a blue sky with clouds, surrounded by brick walls and gravel, located in Halesworth, Suffolk, UK.
Keywords: The Cut, Halesworth, Suffolk, England, UK, Europe, Arts centre, Red Brick Building, Brickwork, Blue sky, Clouds, Gravel, Industrial, Mill, Sign, Bushes<|end_of_box|>
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
- *Observations:* Required fields are missing or empty; Extra text appears
  before the Title field; Response appears cut off at the token limit;
  Internal reasoning block appears incomplete; Internal reasoning text remains
  visible
- *Model load time:* 3.21s
- *Generation time:* 109.77s
- *Total time:* 113.41s
- *Input validation time:* 0.413
- *Prompt preparation time:* 0.00156
- *First-token latency:* 1.53
- *Cleanup time:* 0.157
- *Prompt tokens:* 1,308
- *Generation tokens:* 500
- *Total tokens:* 1,808
- *Prompt throughput (raw):* 854 tok/s
- *Generation throughput (raw):* 4.65 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.6
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00533
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1420
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
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a title, a descriptive paragraph, and 10-18 keywords.

Starting with the title: The image shows a large red-brick building with "THE CUT" on it, located in Halesworth, Suffolk, UK. The setting is a town centre, so maybe something like "Red-brick building 'The Cut' in Halesworth, Suffolk, UK" but needs to be concise. Maybe "Red-brick building 'The Cut' in Halesworth, Suffolk, UK" or "Halesworth's Red Brick Building 'The Cut'". Wait, the title hint mentions "Town centre, Halesworth, England, UK, GBR, Europe" so including that. Maybe "Red-brick building 'The Cut' in Halesworth town centre, Suffolk, UK".

For the description: Need to combine context and visible details. The building is a large red-brick structure with multiple windows, a white section with "THE CUT" sign, brick walls, gravel ground, bushes, blue sky with clouds. So something like "A large red-brick building with multiple windows and a white section displaying 'THE CUT' stands in Halesworth town centre, surrounded by brick walls, gravel, and greenery under a partly cloudy blue sky."

Keywords: Let's list the visible elements. Red-brick building, Halesworth, England, UK, Europe, Town centre, The Cut, brickwork, brick walls, gravel, bushes, blue sky, clouds, industrial, locations, mill (maybe, since it's a large brick building which could be a mill), red brick building, sky, trees, satellite dish, satellite dish on the building, white section, sign, windows, industrial architecture. Wait, the keyword hints include "Industrial" and "Mill" as hints. So include those. Also, "Adobe Stock" and "Any Vision" are hints but maybe not visible. The user said to omit uncertain details. So focus on visible elements. Let's check the image: brickwork, brick walls, gravel, bushes, blue sky, clouds, red brick building, The Cut sign, industrial, Halesworth, England, UK, Europe, town centre, windows, satellite dish, trees, satellite dish on building, white section, sign. That's 15 keywords.
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a title, a descriptive paragraph, and 10-18 keywords. 

Starting with the title: The image shows a large red-brick building with "THE CUT" on it, located in Halesworth, Suffolk, UK. The setting is a town centre, so maybe something like "Red-brick building 'The Cut' in Halesworth, Suffolk, UK" but needs to be concise. Maybe "Red-brick building 'The Cut' in Halesworth, Suffolk, UK" or "Halesworth's Red Brick Building 'The Cut'". Wait, the title hint mentions "Town centre, Halesworth, England, UK, GBR, Europe" so including that. Maybe "Red-brick building 'The Cut' in Halesworth town centre, Suffolk, UK".

For the description: Need to combine context and visible details. The building is a large red-brick structure with multiple windows, a white section with "THE CUT" sign, brick walls, gravel ground, bushes, blue sky with clouds. So something like "A large red-brick building with multiple windows and a white section displaying 'THE CUT' stands in Halesworth town centre, surrounded by brick walls, gravel, and greenery under a partly cloudy blue sky."

Keywords: Let's list the visible elements. Red-brick building, Halesworth, England, UK, Europe, Town centre, The Cut, brickwork, brick walls, gravel, bushes, blue sky, clouds, industrial, locations, mill (maybe, since it's a large brick building which could be a mill), red brick building, sky, trees, satellite dish, satellite dish on the building, white section, sign, windows, industrial architecture. Wait, the keyword hints include "Industrial" and "Mill" as hints. So include those. Also, "Adobe Stock" and "Any Vision" are hints but maybe not visible. The user said to omit uncertain details. So focus on visible elements. Let's check the image: brickwork, brick walls, gravel, bushes, blue sky, clouds, red brick building, The Cut sign, industrial, Halesworth, England, UK, Europe, town centre, windows, satellite dish, trees, satellite dish on building, white section, sign. That's 15 keywords.
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
- *Observations:* Required fields are missing or empty; Extra text appears
  before the Title field; Response appears cut off at the token limit;
  Internal reasoning block appears incomplete; Internal reasoning text remains
  visible
- *Model load time:* 2.29s
- *Generation time:* 9.19s
- *Total time:* 11.90s
- *Input validation time:* 0.416
- *Prompt preparation time:* 0.0016
- *First-token latency:* 0.874
- *Cleanup time:* 0.101
- *Prompt tokens:* 1,308
- *Generation tokens:* 500
- *Total tokens:* 1,808
- *Prompt throughput (raw):* 1,497 tok/s
- *Generation throughput (raw):* 65.8 tok/s
- *Peak memory:* 22
- *Active memory:* 18
- *Cache memory:* 1.3
- *Model-load active memory:* 17.8
- *Post-cleanup active memory:* 0.00598
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1420
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
◁think▷Okay, let's tackle this query. The user wants me to create British-English catalogue metadata based on the image and the provided context. First, I need to parse the instructions carefully. The key points are:

1. **Title**: A concrete 5-10 word title.
2. **Description**: 1-2 sentences combining context and visible details, including setting, action, lighting, etc.
3. **Keywords**: 10-18 unique keywords, covering context and visible details.

The context given includes the capture date/time (2026-08-01 16:28:40 UTC+01:00) and GPS coordinates (52.345200°N, 1.503700°E). The location is Halesworth, Suffolk, UK. The image shows a brick building labeled "THE CUT" with a blue sky, clouds, bushes, gravel, etc.

First, the title should be concise. The building is a red brick building with "THE CUT" on it. Maybe "Red-Brick Building 'The Cut' in Halesworth, Suffolk" – that's 8 words. Good.

For the description, I need to combine the context and visible details. The main subject is the red brick building, part of the Cut in Halesworth. The setting is town centre, so I should mention that. The sky is blue with clouds, there's a gravel area, bushes, and the building has a sign. The building is an industrial structure, possibly a former mill. The lighting is daytime. So something like "A red brick building labeled 'The Cut' stands prominently in Halesworth's town centre, Suffolk, under a partly cloudy sky with gravel surroundings and lush greenery." That covers the main elements.

Keywords: Red brick building, THE CUT, Halesworth, Suffolk, UK, Europe, blue sky, clouds, bushes, gravel, brickwork, industrial architecture, town centre, historic mill conversion, signage, daylight. Let me check if these are all visible in the image. The image shows brickwork, the sign "THE CUT", the location in Halesworth, the sky, bushes, gravel, and it's daytime. Industrial architecture and historic mill conversion might be inferred from the building's style. Signage is visible. That should cover it. Need to make sure there are exactly 10-18 keywords. Let me count: 1. Red brick building 2. THE CUT 3
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this query. The user wants me to create British-English catalogue metadata based on the image and the provided context. First, I need to parse the instructions carefully. The key points are:

1. **Title**: A concrete 5-10 word title.
2. **Description**: 1-2 sentences combining context and visible details, including setting, action, lighting, etc.
3. **Keywords**: 10-18 unique keywords, covering context and visible details.

The context given includes the capture date/time (2026-08-01 16:28:40 UTC+01:00) and GPS coordinates (52.345200°N, 1.503700°E). The location is Halesworth, Suffolk, UK. The image shows a brick building labeled "THE CUT" with a blue sky, clouds, bushes, gravel, etc.

First, the title should be concise. The building is a red brick building with "THE CUT" on it. Maybe "Red-Brick Building 'The Cut' in Halesworth, Suffolk" – that's 8 words. Good.

For the description, I need to combine the context and visible details. The main subject is the red brick building, part of the Cut in Halesworth. The setting is town centre, so I should mention that. The sky is blue with clouds, there's a gravel area, bushes, and the building has a sign. The building is an industrial structure, possibly a former mill. The lighting is daytime. So something like "A red brick building labeled 'The Cut' stands prominently in Halesworth's town centre, Suffolk, under a partly cloudy sky with gravel surroundings and lush greenery." That covers the main elements.

Keywords: Red brick building, THE CUT, Halesworth, Suffolk, UK, Europe, blue sky, clouds, bushes, gravel, brickwork, industrial architecture, town centre, historic mill conversion, signage, daylight. Let me check if these are all visible in the image. The image shows brickwork, the sign "THE CUT", the location in Halesworth, the sky, bushes, gravel, and it's daytime. Industrial architecture and historic mill conversion might be inferred from the building's style. Signage is visible. That should cover it. Need to make sure there are exactly 10-18 keywords. Let me count: 1. Red brick building 2. THE CUT 3
```

</details>

</details>

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### mlx-community/LFM2-VL-1.6B-8bit

<details>
<summary>Complete evidence: mlx-community/LFM2-VL-1.6B-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit
- *Model load time:* 0.61s
- *Generation time:* 2.14s
- *Total time:* 3.16s
- *Input validation time:* 0.414
- *Prompt preparation time:* 0.00119
- *First-token latency:* 0.0946
- *Cleanup time:* 0.0876
- *Prompt tokens:* 580
- *Generation tokens:* 500
- *Total tokens:* 1,080
- *Prompt throughput (raw):* 6,133 tok/s
- *Generation throughput (raw):* 327 tok/s
- *Peak memory:* 3.0
- *Active memory:* 2.1
- *Cache memory:* 0.04
- *Model-load active memory:* 2.06
- *Post-cleanup active memory:* 0.00611
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1284
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
Title: The Cut, Halesworth, Suffolk, England, UK, GBR, Europe

Description: A red brick building with a white sign reading "The Cut" stands prominently in the town center of Halesworth, Suffolk, England. The building, likely an industrial or historical structure, features a series of windows and a flat roof. The surrounding area includes a brick wall and some greenery, with a clear blue sky overhead.

Keywords: Halesworth, Suffolk, England, UK, GBR, Europe, red brick, industrial building, town center, white sign, windows, brick wall, greenery, blue sky, England, Europe, industrial, historical, Halesworth, Suffolk, England, UK, GBR, Europe, The Cut, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut, Halesworth, Suffolk, England, UK, GBR, Europe

Description: A red brick building with a white sign reading "The Cut" stands prominently in the town center of Halesworth, Suffolk, England. The building, likely an industrial or historical structure, features a series of windows and a flat roof. The surrounding area includes a brick wall and some greenery, with a clear blue sky overhead.

Keywords: Halesworth, Suffolk, England, UK, GBR, Europe, red brick, industrial building, town center, white sign, windows, brick wall, greenery, blue sky, England, Europe, industrial, historical, Halesworth, Suffolk, England, UK, GBR, Europe, The Cut, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR,
```

</details>

</details>

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### mlx-community/MiniCPM-V-4.6-8bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.6-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty; Extra text appears
  before the Title field
- *Model load time:* 0.87s
- *Generation time:* 1.54s
- *Total time:* 2.86s
- *Input validation time:* 0.425
- *Prompt preparation time:* 0.00846
- *First-token latency:* 0.631
- *Cleanup time:* 0.0803
- *Prompt tokens:* 918
- *Generation tokens:* 55
- *Total tokens:* 973
- *Prompt throughput (raw):* 1,455 tok/s
- *Generation throughput (raw):* 272 tok/s
- *Peak memory:* 3.8
- *Active memory:* 2.3
- *Cache memory:* 0.05
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00729
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1301
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
Industrial brick building with modern sign in town center.

Keywords: brickwork, industrial, mill, red brick, building, sign, sky, clouds, England, Europe, Suffolk, Gravel, bushes, roof, trees, Car, Clouds, England, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
Industrial brick building with modern sign in town center.

Keywords: brickwork, industrial, mill, red brick, building, sign, sky, clouds, England, Europe, Suffolk, Gravel, bushes, roof, trees, Car, Clouds, England, Europe
```

</details>

</details>

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### mlx-community/Molmo-7B-D-0924-8bit

<details>
<summary>Complete evidence: mlx-community/Molmo-7B-D-0924-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* No response text was returned; Required fields are missing
  or empty
- *Model load time:* 1.17s
- *Generation time:* 18.95s
- *Total time:* 20.55s
- *Input validation time:* 0.413
- *Prompt preparation time:* 0.000148
- *First-token latency:* 17.9
- *Cleanup time:* 0.144
- *Prompt tokens:* 1,506
- *Generation tokens:* 1
- *Total tokens:* 1,507
- *Prompt throughput (raw):* 84.0 tok/s
- *Generation throughput (raw):* 35,140 tok/s
- *Peak memory:* 41
- *Active memory:* 9.0
- *Cache memory:* 0.13
- *Model-load active memory:* 9.04
- *Post-cleanup active memory:* 0.00839
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1212
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

empty output

</details>

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### mlx-community/Molmo-7B-D-0924-bf16

<details>
<summary>Complete evidence: mlx-community/Molmo-7B-D-0924-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* No response text was returned; Required fields are missing
  or empty
- *Model load time:* 1.71s
- *Generation time:* 18.45s
- *Total time:* 20.58s
- *Input validation time:* 0.412
- *Prompt preparation time:* 0.000117
- *First-token latency:* 17.4
- *Cleanup time:* 0.16
- *Prompt tokens:* 1,506
- *Generation tokens:* 1
- *Total tokens:* 1,507
- *Prompt throughput (raw):* 86.5 tok/s
- *Generation throughput (raw):* 52,980 tok/s
- *Peak memory:* 48
- *Active memory:* 16
- *Cache memory:* 0.12
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0087
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1212
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

empty output

</details>

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### mlx-community/MolmoPoint-8B-fp16

<details>
<summary>Complete evidence: mlx-community/MolmoPoint-8B-fp16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty
- *Model load time:* 2.25s
- *Generation time:* 22.71s
- *Total time:* 25.38s
- *Input validation time:* 0.411
- *Prompt preparation time:* 0.00499
- *First-token latency:* 2.67
- *Cleanup time:* 0.116
- *Prompt tokens:* 3,123
- *Generation tokens:* 115
- *Total tokens:* 3,238
- *Prompt throughput (raw):* 1,171 tok/s
- *Generation throughput (raw):* 6 tok/s
- *Peak memory:* 26
- *Active memory:* 19
- *Cache memory:* 5.0
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 0.00932
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1271
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
The Cut Arts Centre, Halesworth, Suffolk, UK

The Cut arts centre stands in Halesworth, Suffolk, UK, featuring a red brick building with a white side and a sign reading "The Cut" and "www.the-cut.org.uk". The structure is surrounded by a brick wall with bushes and trees, set against a blue sky with white clouds.

Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk
</pre>

<details>
<summary>Exact raw output</summary>

```text
The Cut Arts Centre, Halesworth, Suffolk, UK

The Cut arts centre stands in Halesworth, Suffolk, UK, featuring a red brick building with a white side and a sign reading "The Cut" and "www.the-cut.org.uk". The structure is surrounded by a brick wall with bushes and trees, set against a blue sky with white clouds.

Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk
```

</details>

</details>

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### mlx-community/Qwen2-VL-2B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen2-VL-2B-Instruct-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty
- *Model load time:* 0.58s
- *Generation time:* 79.41s
- *Total time:* 80.53s
- *Input validation time:* 0.529
- *Prompt preparation time:* 0.00426
- *First-token latency:* 77.6
- *Cleanup time:* 0.107
- *Prompt tokens:* 16,592
- *Generation tokens:* 128
- *Total tokens:* 16,720
- *Prompt throughput (raw):* 214 tok/s
- *Generation throughput (raw):* 200 tok/s
- *Peak memory:* 5.1
- *Active memory:* 1.3
- *Cache memory:* 0.86
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1363
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
Title:
The Cut, Halesworth, Suffolk, UK, GBR, Europe

Description:
The Cut is a town centre arts centre located in Halesworth, Suffolk, UK. The building is a large, multi-story brick structure with a prominent sign that reads "The Cut" and a website address. The building is surrounded by a brick wall and has a blue sky with white clouds in the background. The building features large windows and a red brick facade, and there are bushes and plants growing around the building. The building is part of the Halesworth industrial complex and is a popular destination for arts and culture events.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
The Cut, Halesworth, Suffolk, UK, GBR, Europe

Description:
The Cut is a town centre arts centre located in Halesworth, Suffolk, UK. The building is a large, multi-story brick structure with a prominent sign that reads "The Cut" and a website address. The building is surrounded by a brick wall and has a blue sky with white clouds in the background. The building features large windows and a red brick facade, and there are bushes and plants growing around the building. The building is part of the Halesworth industrial complex and is a popular destination for arts and culture events.
```

</details>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty; Response appears cut
  off at the token limit
- *Model load time:* 0.89s
- *Generation time:* 27.90s
- *Total time:* 29.29s
- *Input validation time:* 0.486
- *Prompt preparation time:* 0.0139
- *First-token latency:* 20.5
- *Cleanup time:* 0.211
- *Prompt tokens:* 16,583
- *Generation tokens:* 500
- *Total tokens:* 17,083
- *Prompt throughput (raw):* 809 tok/s
- *Generation throughput (raw):* 78.8 tok/s
- *Peak memory:* 8.6
- *Active memory:* 4.3
- *Cache memory:* 2.4
- *Model-load active memory:* 4.27
- *Post-cleanup active memory:* 0.0108
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1313
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
Got it, let's tackle this step by step. First, the title. The context says "The Cut in Halesworth, Suffolk in the UK" and the image shows a red brick building with "THE CUT" sign. So the title should be concise. Maybe "The Cut, Halesworth, Suffolk, UK" but need to check if it's 5-10 words. Let's see: "The Cut, Halesworth, Suffolk, UK" is 7 words. That's good.

Next, the description. Need to combine context with visible subject, setting, action, lighting, details. The image has a red brick building (The Cut), blue sky with clouds, brickwork, bushes, gravel ground. The building is industrial, maybe a mill. The capture date is 2026-08-01, but we don't claim that. The description should mention the building, location (Halesworth, Suffolk), the sign "THE CUT", the sky, and the setting. Let's structure: "A red brick industrial building labeled 'THE CUT' in Halesworth, Suffolk, UK, under a bright blue sky with scattered clouds, featuring brickwork, bushes, and a gravel foreground."

Wait, check the context: "The Cut in Halesworth, Suffolk in the UK" is the description hint. So the description should include that. Also, the building is a red brick mill? The context says "Mill" is a keyword. So "A red brick mill building labeled 'THE CUT' in Halesworth, Suffolk, UK, under a bright blue sky with scattered clouds, featuring brickwork, bushes, and a gravel foreground."

Now keywords. Need 10-18 unique, comma-separated. From context: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk. But we need to pick relevant ones. The image has: Red Brick Building, Roof (the roof is dark, maybe slate), Sign (THE CUT), Bushes, Clouds, Sky, England, Europe, Suffolk, Industrial, Mill, Brickwork, Gravel, Blue sky. Let's list unique ones. Let's check the image: the building is red brick, so "Red Brick Building" is
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this step by step. First, the title. The context says "The Cut in Halesworth, Suffolk in the UK" and the image shows a red brick building with "THE CUT" sign. So the title should be concise. Maybe "The Cut, Halesworth, Suffolk, UK" but need to check if it's 5-10 words. Let's see: "The Cut, Halesworth, Suffolk, UK" is 7 words. That's good.

Next, the description. Need to combine context with visible subject, setting, action, lighting, details. The image has a red brick building (The Cut), blue sky with clouds, brickwork, bushes, gravel ground. The building is industrial, maybe a mill. The capture date is 2026-08-01, but we don't claim that. The description should mention the building, location (Halesworth, Suffolk), the sign "THE CUT", the sky, and the setting. Let's structure: "A red brick industrial building labeled 'THE CUT' in Halesworth, Suffolk, UK, under a bright blue sky with scattered clouds, featuring brickwork, bushes, and a gravel foreground."

Wait, check the context: "The Cut in Halesworth, Suffolk in the UK" is the description hint. So the description should include that. Also, the building is a red brick mill? The context says "Mill" is a keyword. So "A red brick mill building labeled 'THE CUT' in Halesworth, Suffolk, UK, under a bright blue sky with scattered clouds, featuring brickwork, bushes, and a gravel foreground."

Now keywords. Need 10-18 unique, comma-separated. From context: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk. But we need to pick relevant ones. The image has: Red Brick Building, Roof (the roof is dark, maybe slate), Sign (THE CUT), Bushes, Clouds, Sky, England, Europe, Suffolk, Industrial, Mill, Brickwork, Gravel, Blue sky. Let's list unique ones. Let's check the image: the building is red brick, so "Red Brick Building" is
```

</details>

</details>

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### mlx-community/SmolVLM-Instruct-bf16

<details>
<summary>Complete evidence: mlx-community/SmolVLM-Instruct-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty
- *Model load time:* 0.90s
- *Generation time:* 1.27s
- *Total time:* 2.97s
- *Input validation time:* 0.793
- *Prompt preparation time:* 0.00122
- *First-token latency:* 0.4
- *Cleanup time:* 0.0763
- *Prompt tokens:* 1,522
- *Generation tokens:* 13
- *Total tokens:* 1,535
- *Prompt throughput (raw):* 3,805 tok/s
- *Generation throughput (raw):* 124 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.35
- *Model-load active memory:* 4.51
- *Post-cleanup active memory:* 0.0145
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1265
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
 The Cut in Halesworth, Suffolk in the UK.
</pre>

<details>
<summary>Exact raw output</summary>

```text
 The Cut in Halesworth, Suffolk in the UK.
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
  the token limit
- *Model load time:* 7.61s
- *Generation time:* 36.14s
- *Total time:* 44.18s
- *Input validation time:* 0.416
- *Prompt preparation time:* 0.000272
- *First-token latency:* 24.6
- *Cleanup time:* 0.184
- *Prompt tokens:* 16,592
- *Generation tokens:* 500
- *Total tokens:* 17,092
- *Prompt throughput (raw):* 674 tok/s
- *Generation throughput (raw):* 47.5 tok/s
- *Peak memory:* 14
- *Active memory:* 9.5
- *Cache memory:* 1.3
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0148
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1363
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
Title: The Cut in Halesworth, Suffolk

Description: A historic red brick building with "The Cut" signage stands under a bright blue sky, showcasing industrial architecture in Halesworth, Suffolk, UK.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Europe, Red brick building, Industrial, Arts centre, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut in Halesworth, Suffolk

Description: A historic red brick building with "The Cut" signage stands under a bright blue sky, showcasing industrial architecture in Halesworth, Suffolk, UK.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Europe, Red brick building, Industrial, Arts centre, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic,
```

</details>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible; Required
  fields are missing or empty; Extra text appears before the Title field
- *Model load time:* 3.49s
- *Generation time:* 2.89s
- *Total time:* 7.22s
- *Input validation time:* 0.799
- *Prompt preparation time:* 0.0261
- *First-token latency:* 0.37
- *Cleanup time:* 0.137
- *Prompt tokens:* 605
- *Generation tokens:* 85
- *Total tokens:* 690
- *Prompt throughput (raw):* 1,635 tok/s
- *Generation throughput (raw):* 51.9 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0148
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1259
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
&lt;channel|&gt;Title: The Cut Arts Centre in Halesworth, Suffolk
Description: A large multi-story red brick industrial building repurposed as an arts centre stands in Halesworth, England, under a partly cloudy sky.
Keywords: Halesworth, Suffolk, England, red brick, industrial, arts centre, mill, architecture, brickwork, clouds, sky, windows, sign, historic, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: The Cut Arts Centre in Halesworth, Suffolk
Description: A large multi-story red brick industrial building repurposed as an arts centre stands in Halesworth, England, under a partly cloudy sky.
Keywords: Halesworth, Suffolk, England, red brick, industrial, arts centre, mill, architecture, brickwork, clouds, sky, windows, sign, historic, Europe
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
- *Observations:* Unrecognised model control tokens remain visible; Required
  fields are missing or empty; Extra text appears before the Title field
- *Model load time:* 3.33s
- *Generation time:* 2.55s
- *Total time:* 6.31s
- *Input validation time:* 0.422
- *Prompt preparation time:* 0.000281
- *First-token latency:* 0.37
- *Cleanup time:* 0.193
- *Prompt tokens:* 605
- *Generation tokens:* 86
- *Total tokens:* 691
- *Prompt throughput (raw):* 1,634 tok/s
- *Generation throughput (raw):* 70.9 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0148
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1259
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
&lt;channel|&gt;Title: The Cut Arts Centre Building in Halesworth, Suffolk
Description: A large multi-story red brick industrial building serves as The Cut arts centre, located in Halesworth under a partly cloudy sky.
Keywords: Halesworth, Suffolk, England, industrial, red brick, mill, arts much, architecture, building, sky, clouds, windows, sign, wall, historic, urban
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: The Cut Arts Centre Building in Halesworth, Suffolk
Description: A large multi-story red brick industrial building serves as The Cut arts centre, located in Halesworth under a partly cloudy sky.
Keywords: Halesworth, Suffolk, England, industrial, red brick, mill, arts much, architecture, building, sky, clouds, windows, sign, wall, historic, urban
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
- *Observations:* Required fields are missing or empty; Response appears cut
  off at the token limit
- *Model load time:* 1.30s
- *Generation time:* 4.90s
- *Total time:* 6.63s
- *Input validation time:* 0.416
- *Prompt preparation time:* 0.000182
- *First-token latency:* 0.293
- *Cleanup time:* 0.0996
- *Prompt tokens:* 582
- *Generation tokens:* 500
- *Total tokens:* 1,082
- *Prompt throughput (raw):* 1,988 tok/s
- *Generation throughput (raw):* 124 tok/s
- *Peak memory:* 6.0
- *Active memory:* 4.5
- *Cache memory:* 0.04
- *Model-load active memory:* 4.48
- *Post-cleanup active memory:* 0.0165
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1231
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

- 10-18-10-18-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1
</pre>

<details>
<summary>Exact raw output</summary>

```text

- 10-18-10-18-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1
```

</details>

</details>

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### mlx-community/gemma-3n-E4B-it-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E4B-it-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty; Extra text appears
  before the Title field
- *Model load time:* 2.11s
- *Generation time:* 3.20s
- *Total time:* 5.74s
- *Input validation time:* 0.415
- *Prompt preparation time:* 0.00333
- *First-token latency:* 0.399
- *Cleanup time:* 0.13
- *Prompt tokens:* 590
- *Generation tokens:* 104
- *Total tokens:* 694
- *Prompt throughput (raw):* 1,478 tok/s
- *Generation throughput (raw):* 46.6 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.017
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1290
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
The image shows a large, red brick building in Halesworth, Suffolk, UK. The building has multiple windows and a white sign that reads "THE CUT" in large letters. The building is surrounded by brick walls and some greenery. The sky is blue with some white clouds.

**Keywords:**

* Halesworth
* Suffolk
* UK
* The Cut
* Red brick building
* Windows
* White sign
* Brick walls
* Greenery
* Blue sky
* Clouds
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows a large, red brick building in Halesworth, Suffolk, UK. The building has multiple windows and a white sign that reads "THE CUT" in large letters. The building is surrounded by brick walls and some greenery. The sky is blue with some white clouds.

**Keywords:**

* Halesworth
* Suffolk
* UK
* The Cut
* Red brick building
* Windows
* White sign
* Brick walls
* Greenery
* Blue sky
* Clouds
```

</details>

</details>

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### mlx-community/gemma-4-31b-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-4-31b-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* No response text was returned; Required fields are missing
  or empty
- *Model load time:* 11.24s
- *Generation time:* 8.87s
- *Total time:* 20.53s
- *Input validation time:* 0.414
- *Prompt preparation time:* 0.000145
- *First-token latency:* 8.19
- *Cleanup time:* 0.28
- *Prompt tokens:* 597
- *Generation tokens:* 1
- *Total tokens:* 598
- *Prompt throughput (raw):* 72.9 tok/s
- *Generation throughput (raw):* 39,216 tok/s
- *Peak memory:* 64
- *Active memory:* 63
- *Cache memory:* 0.79
- *Model-load active memory:* 62.6
- *Post-cleanup active memory:* 0.018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1222
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

empty output

</details>

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### mlx-community/llava-v1.6-mistral-7b-8bit

<details>
<summary>Complete evidence: mlx-community/llava-v1.6-mistral-7b-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty
- *Model load time:* 0.92s
- *Generation time:* 5.76s
- *Total time:* 7.11s
- *Input validation time:* 0.426
- *Prompt preparation time:* 0.00193
- *First-token latency:* 2.6
- *Cleanup time:* 0.105
- *Prompt tokens:* 2,413
- *Generation tokens:* 133
- *Total tokens:* 2,546
- *Prompt throughput (raw):* 928 tok/s
- *Generation throughput (raw):* 57.3 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.90
- *Model-load active memory:* 8.06
- *Post-cleanup active memory:* 0.0186
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1235
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
The Cut, Halesworth, Suffolk, England, UK, GBR, Europe, 2026-08-01 16:28:40 UTC+01:00, 52.345200°N, 1.503700°E, Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk
</pre>

<details>
<summary>Exact raw output</summary>

```text
The Cut, Halesworth, Suffolk, England, UK, GBR, Europe, 2026-08-01 16:28:40 UTC+01:00, 52.345200°N, 1.503700°E, Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk 
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
- *Observations:* Required fields are missing or empty
- *Model load time:* 0.49s
- *Generation time:* 1.07s
- *Total time:* 1.98s
- *Input validation time:* 0.423
- *Prompt preparation time:* 0.00141
- *First-token latency:* 0.0822
- *Cleanup time:* 0.0816
- *Prompt tokens:* 327
- *Generation tokens:* 121
- *Total tokens:* 448
- *Prompt throughput (raw):* 3,977 tok/s
- *Generation throughput (raw):* 345 tok/s
- *Peak memory:* 2.3
- *Active memory:* 0.62
- *Cache memory:* 0.30
- *Model-load active memory:* 0.617
- *Post-cleanup active memory:* 0.0193
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1319
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
Title: The Cut in Halesworth, England, UK, GBR, Europe
Description: The Brickwork in Halesworth, Suffolk in the UK, featuring the Cut in the town center. The image captures the essence of the town's historic architecture, with the red brick building standing out amidst the greenery. The image is a blend of the old and the new, with the red brick building being the focal point, while the greenery adds a touch of nature to the scene. The image is a testament to the town's rich history and its blend of old and new architecture.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut in Halesworth, England, UK, GBR, Europe
Description: The Brickwork in Halesworth, Suffolk in the UK, featuring the Cut in the town center. The image captures the essence of the town's historic architecture, with the red brick building standing out amidst the greenery. The image is a blend of the old and the new, with the red brick building being the focal point, while the greenery adds a touch of nature to the scene. The image is a testament to the town's rich history and its blend of old and new architecture.
```

</details>

</details>

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### mlx-community/paligemma2-10b-ft-docci-448-6bit

<details>
<summary>Complete evidence: mlx-community/paligemma2-10b-ft-docci-448-6bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty
- *Model load time:* 1.82s
- *Generation time:* 2.08s
- *Total time:* 4.33s
- *Input validation time:* 0.42
- *Prompt preparation time:* 0.000188
- *First-token latency:* 1.03
- *Cleanup time:* 0.12
- *Prompt tokens:* 1,343
- *Generation tokens:* 12
- *Total tokens:* 1,355
- *Prompt throughput (raw):* 1,301 tok/s
- *Generation throughput (raw):* 32.3 tok/s
- *Peak memory:* 12
- *Active memory:* 7.9
- *Cache memory:* 1.7
- *Model-load active memory:* 7.88
- *Post-cleanup active memory:* 0.0203
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1219
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
Write the text in the same font as the image.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Write the text in the same font as the image.
```

</details>

</details>

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### mlx-community/paligemma2-10b-ft-docci-448-bf16

<details>
<summary>Complete evidence: mlx-community/paligemma2-10b-ft-docci-448-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty
- *Model load time:* 2.71s
- *Generation time:* 3.95s
- *Total time:* 7.08s
- *Input validation time:* 0.422
- *Prompt preparation time:* 0.000135
- *First-token latency:* 1.43
- *Cleanup time:* 0.184
- *Prompt tokens:* 1,343
- *Generation tokens:* 10
- *Total tokens:* 1,353
- *Prompt throughput (raw):* 937 tok/s
- *Generation throughput (raw):* 5.55 tok/s
- *Peak memory:* 26
- *Active memory:* 19
- *Cache memory:* 6.2
- *Model-load active memory:* 19.3
- *Post-cleanup active memory:* 0.0213
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1219
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
Write the text in a plain text editor.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Write the text in a plain text editor.
```

</details>

</details>

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### mlx-community/paligemma2-3b-ft-docci-448-bf16

<details>
<summary>Complete evidence: mlx-community/paligemma2-3b-ft-docci-448-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty
- *Model load time:* 1.61s
- *Generation time:* 9.21s
- *Total time:* 11.32s
- *Input validation time:* 0.492
- *Prompt preparation time:* 0.000127
- *First-token latency:* 0.46
- *Cleanup time:* 0.106
- *Prompt tokens:* 1,343
- *Generation tokens:* 150
- *Total tokens:* 1,493
- *Prompt throughput (raw):* 2,922 tok/s
- *Generation throughput (raw):* 18.6 tok/s
- *Peak memory:* 11
- *Active memory:* 6.1
- *Cache memory:* 3.7
- *Model-load active memory:* 6.09
- *Post-cleanup active memory:* 0.0223
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1219
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
A three-quarters view of a red brick building with a pointed roof. The building is in the center of the image, and the side of the building facing the bottom right corner of the image is painted white. There is a gray concrete wall extending across the bottom of the image. There is a tree on the left side of the image, and there is a tree on the right side of the image. There is a gray brick wall extending across the bottom of the image in front of the red brick building. There is a gray cement wall extending across the bottom of the image in the foreground. There is a tree on the left side of the image in front of the gray cement wall. The sky is bright and filled with white clouds.
</pre>

<details>
<summary>Exact raw output</summary>

```text
A three-quarters view of a red brick building with a pointed roof. The building is in the center of the image, and the side of the building facing the bottom right corner of the image is painted white. There is a gray concrete wall extending across the bottom of the image. There is a tree on the left side of the image, and there is a tree on the right side of the image. There is a gray brick wall extending across the bottom of the image in front of the red brick building. There is a gray cement wall extending across the bottom of the image in the foreground. There is a tree on the left side of the image in front of the gray cement wall. The sky is bright and filled with white clouds.
```

</details>

</details>

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### mlx-community/paligemma2-3b-pt-896-4bit

<details>
<summary>Complete evidence: mlx-community/paligemma2-3b-pt-896-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Required fields are missing
  or empty; Response repeats the task instructions instead of only returning
  the requested fields; Response appears cut off at the token limit
- *Model load time:* 1.43s
- *Generation time:* 18.17s
- *Total time:* 20.04s
- *Input validation time:* 0.421
- *Prompt preparation time:* 0.000135
- *First-token latency:* 1.36
- *Cleanup time:* 0.272
- *Prompt tokens:* 4,415
- *Generation tokens:* 500
- *Total tokens:* 4,915
- *Prompt throughput (raw):* 3,236 tok/s
- *Generation throughput (raw):* 31.6 tok/s
- *Peak memory:* 4.6
- *Active memory:* 1.7
- *Cache memory:* 1.7
- *Model-load active memory:* 1.74
- *Post-cleanup active memory:* 0.0234
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1219
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
- Title hint: Town centre, Halesworth, England, UK, GBR, Europe
- Description hint: The Cut in Halesworth, Suffolk in the UK
- Keyword hints: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context
</pre>

<details>
<summary>Exact raw output</summary>

```text
- Title hint: Town centre, Halesworth, England, UK, GBR, Europe
- Description hint: The Cut in Halesworth, Suffolk in the UK
- Keyword hints: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context
```

</details>

</details>

---

<a id="model-qnguyen3-nanollava"></a>

### qnguyen3/nanoLLaVA

<details>
<summary>Complete evidence: qnguyen3/nanoLLaVA</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Required fields are missing or empty
- *Model load time:* 0.56s
- *Generation time:* 1.28s
- *Total time:* 2.27s
- *Input validation time:* 0.425
- *Prompt preparation time:* 0.000582
- *First-token latency:* 0.0925
- *Cleanup time:* 0.108
- *Prompt tokens:* 327
- *Generation tokens:* 62
- *Total tokens:* 389
- *Prompt throughput (raw):* 3,534 tok/s
- *Generation throughput (raw):* 111 tok/s
- *Peak memory:* 4.6
- *Active memory:* 2.1
- *Cache memory:* 1.6
- *Model-load active memory:* 2.12
- *Post-cleanup active memory:* 0.0245
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1319
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
Title: The Cut in Halesworth, England, UK, GBR, Europe
Description: A red brick building in Halesworth, Suffolk in the UK, with a distinctive red brick chimney and a large window. The building is surrounded by a white brick wall and has a sign on it.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: The Cut in Halesworth, England, UK, GBR, Europe
Description: A red brick building in Halesworth, Suffolk in the UK, with a distinctive red brick chimney and a large window. The building is surrounded by a white brick wall and has a sign on it.
```

</details>

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
- *Total time:* 0.45s
- *Input validation time:* 0.423
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0647
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.0145
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11500, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 10864, in _load_model
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
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 955, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 590, in get_model_path
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
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 268, in snapshot_download
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3643, in repo_info
    return method(
        repo_id,
    ...<4 lines>...
        files_metadata=files_metadata,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3322, in model_info
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11753, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11515, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: [Errno 54] Connection reset by peer

```

*Captured upstream output:*

```text
=== STDERR ===
[01:10:15] DEBUG    HF Cache Info for mlx-community/SmolVLM2-2.2B-Instruct-mlx: size=4290.2 MB, files=12
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
- *Model load time:* 11.01s
- *Generation time:* -
- *Total time:* 12.22s
- *Input validation time:* 0.416
- *Prompt preparation time:* 1.17e-05
- *First-token latency:* -
- *Cleanup time:* 1.68
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.0145
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
- *Resolved model revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Step-3.7-Flash-oQ2e/snapshots/3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Generation settings:* not captured
- *EOS token:* not captured

*Complete traceback:*

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11250, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11038, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 10971, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11753, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11519, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11291, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.

```

*Captured upstream output:*

```text
=== STDERR ===
[01:10:27] ERROR    Model preflight validation failed for mlx-community/Step-3.7-Flash-oQ2e
                    ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---
