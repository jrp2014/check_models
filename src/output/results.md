# Model Performance Results

_Generated on 2026-01-23 21:53:39 GMT_

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (386.3 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (2.5 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.00s)
- **📊 Average TPS:** 75.1 across 30 models

## 📈 Resource Usage

- **Total peak memory:** 513.9 GB
- **Average peak memory:** 17.1 GB
- **Memory efficiency:** 199 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 14 | ✅ B: 11 | 🟠 D: 3 | ❌ F: 2

**Average Utility Score:** 76/100

- **Best for cataloging:** `Qwen/Qwen3-VL-2B-Instruct` (🏆 A, 98/100)
- **Worst for cataloging:** `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (❌ F, 1/100)

### ⚠️ 5 Models with Low Utility (D/F)

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (1/100) - Output too short to be useful
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: 🟠 D (50/100) - Lacks visual description of image content
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (43/100) - Lacks visual description of image content
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: 🟠 D (41/100) - Lacks visual description of image content
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (21/100) - Lacks visual description of image content

## ⚠️ Quality Issues

- **❌ Failed Models (8):**
  - `microsoft/Florence-2-large-ft` (`Weight Mismatch`)
  - `mlx-community/InternVL3-14B-8bit` (`Processor Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (`Lib Version`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Lib Version`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (`Config Missing`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (`OOM`)
  - `mlx-community/deepseek-vl2-8bit` (`Type Cast Error`)
  - `prince-canuma/Florence-2-large-ft` (`Model Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "on the right side..."`)
- **👻 Hallucinations (1):**
  - `microsoft/Phi-3.5-vision-instruct`
- **📝 Formatting Issues (3):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 75.1 | Min: 3.84 | Max: 386
- **Peak Memory**: Avg: 17 | Min: 2.5 | Max: 51
- **Total Time**: Avg: 26.39s | Min: 1.99s | Max: 100.54s
- **Generation Time**: Avg: 23.39s | Min: 1.00s | Max: 96.21s
- **Model Load Time**: Avg: 3.00s | Min: 0.99s | Max: 7.38s

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `mlx-vlm` | 5 | Config Missing, Lib Version, Model Error, Processor Error | `mlx-community/InternVL3-14B-8bit`, `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `prince-canuma/Florence-2-large-ft` |
| `mlx` | 3 | OOM, Type Cast Error, Weight Mismatch | `microsoft/Florence-2-large-ft`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/deepseek-vl2-8bit` |

### Actionable Items by Package

#### mlx-vlm

- **mlx-community/InternVL3-14B-8bit** (Processor Error)
  - Error: `Model loading failed: Received a InternVLImageProcessor for argument image_processor, but a ImageProcessingMixin was ...`
  - Type: `ValueError`
- **mlx-community/Kimi-VL-A3B-Thinking-2506-bf16** (Lib Version)
  - Error: `Model loading failed: cannot import name '_validate_images_text_input_order' from 'transformers.processing_utils' (/o...`
  - Type: `ValueError`
- **mlx-community/Kimi-VL-A3B-Thinking-8bit** (Lib Version)
  - Error: `Model loading failed: cannot import name '_validate_images_text_input_order' from 'transformers.processing_utils' (/o...`
  - Type: `ValueError`
- **mlx-community/Phi-3.5-vision-instruct-bf16** (Config Missing)
  - Error: `Model loading failed: /Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots...`
  - Type: `ValueError`
- **prince-canuma/Florence-2-large-ft** (Model Error)
  - Error: `Model loading failed: RobertaTokenizer has no attribute additional_special_tokens`
  - Type: `ValueError`

#### mlx

- **microsoft/Florence-2-large-ft** (Weight Mismatch)
  - Error: `Model loading failed: Missing 1 parameters:
language_model.lm_head.weight.`
  - Type: `ValueError`
- **mlx-community/Qwen2-VL-2B-Instruct-4bit** (OOM)
  - Error: `Model runtime error during generation for mlx-community/Qwen2-VL-2B-Instruct-4bit: [metal::malloc] Attempting to allo...`
  - Type: `ValueError`
- **mlx-community/deepseek-vl2-8bit** (Type Cast Error)
  - Error: `Model runtime error during generation for mlx-community/deepseek-vl2-8bit: std::bad_cast`
  - Type: `ValueError`

> **Prompt used:**
>
> Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.
>
> Context: The image relates to ', Town centre, Wendover, England, United Kingdom, UK
> Here is a caption for the image, crafted according to your specifications:
>
> ***
>
> The cool, quiet light of a late winter afternoon settles over the historic town centre of Wendover, Buckinghamshire, England. This January scene captures The Old House, a distinctive Grade II* listed Georgian residence dating to the early 18th century, standing alongside a row of charming period cottages.
>
> As dusk begins to fall, warm light glows from the classic sash windows, offering a welcoming contrast to the crisp air. The bare, gnarled branches of mature trees frame the view, while the moss-covered clay tiles on the rooftops speak to the buildings' centuries-long history. The scene, with its neat white picket fence and verdant lawn, encapsulates the timeless appeal of a traditional English market town.
> Here is a caption for the image, crafted according to your specifications:
>
> ***
>
> The cool, quiet light of a late winter afternoon settles over the historic town centre of Wendover, Buckinghamshire, England. This January scene captures The Old House, a distinctive Grade II* listed Georgian residence dating to the early 18th century, standing alongside a row of charming period cottages.
>
> As dusk begins to fall, warm light glows from the classic sash windows, offering a welcoming contrast to the crisp air. The bare, gnarled branches of mature trees frame the view, while the moss-covered clay tiles on the rooftops speak to the buildings' centuries-long history. The scene, with its neat white picket fence and verdant lawn, encapsulates the timeless appeal of a traditional English market town.'
>
> The photo was taken around 2026-01-17 16:15:42 GMT  at local time 16:15:42  from GPS 51.764250°N, 0.742750°W . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 818.93s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |            |             |                                   |             mlx |
| `mlx-community/InternVL3-14B-8bit`                      |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |             mlx |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |            |             |                                   |             mlx |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               469 |                   127 |            596 |        2,188 |       386 |         2.5 |            1.00s |      0.99s |       1.99s |                                   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |             2,245 |                    84 |          2,329 |        2,168 |       180 |         4.4 |            1.90s |      1.18s |       3.08s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    | 151,645 |               469 |                   143 |            612 |        1,907 |       103 |         4.7 |            2.12s |      1.12s |       3.23s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,279 |             1,564 |                   123 |          1,687 |        1,283 |       116 |         5.5 |            2.68s |      1.29s |       3.97s |                                   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  49,154 |             1,664 |                   125 |          1,789 |        1,313 |       114 |         5.5 |            2.76s |      1.32s |       4.08s |                                   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |             2,245 |                   161 |          2,406 |          823 |       287 |          51 |            3.79s |      0.99s |       4.78s |                                   |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  49,154 |             1,664 |                   271 |          1,935 |        1,340 |       114 |         5.5 |            4.00s |      1.42s |       5.42s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |       1 |             4,554 |                    22 |          4,576 |        1,196 |       114 |         8.8 |            4.39s |      2.61s |       7.00s | context-ignored                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,482 |                    59 |          1,541 |          502 |      41.3 |          12 |            4.73s |      3.28s |       8.01s | context-ignored                   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             2,958 |                   342 |          3,300 |        1,002 |       156 |         7.3 |            5.49s |      1.78s |       7.27s |                                   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               723 |                   321 |          1,044 |          832 |      41.1 |          17 |            9.04s |      5.14s |      14.18s | verbose                           |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,001 |             2,752 |                   248 |          3,000 |          706 |      28.9 |          19 |           12.88s |      3.87s |      16.75s | formatting                        |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,568 |                    91 |          2,659 |          223 |      42.1 |          12 |           14.05s |      1.90s |      15.96s | context-ignored                   |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,066 |                   349 |          3,415 |          506 |      34.1 |          15 |           16.66s |      3.31s |      19.96s | verbose, bullets(22)              |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               723 |                   387 |          1,110 |          178 |      27.5 |          19 |           18.52s |      5.14s |      23.67s | bullets(30)                       |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,447 |                   245 |          2,692 |          230 |      28.1 |          22 |           19.71s |      3.95s |      23.66s |                                   |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   4,453 |             3,157 |                   500 |          3,657 |          410 |      35.9 |          15 |           21.99s |      3.14s |      25.13s |                                   |                 |
| `mlx-community/pixtral-12b-bf16`                        |       2 |             3,066 |                   391 |          3,457 |          541 |      19.1 |          27 |           26.47s |      5.02s |      31.49s | verbose, bullets(15)              |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |     714 |             1,482 |                   500 |          1,982 |        1,515 |      17.5 |          12 |           29.86s |      3.12s |      32.98s | repetitive(phrase: "on the rig... |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |      82 |             6,480 |                   500 |          6,980 |          296 |      64.2 |          10 |           30.09s |      2.43s |      32.52s | verbose, formatting               |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,482 |                   140 |          1,622 |          412 |      4.92 |          27 |           32.44s |      5.19s |      37.63s | context-ignored                   |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | 115,937 |             6,480 |                   500 |          6,980 |          289 |      46.4 |          13 |           33.55s |      2.79s |      36.34s | verbose, formatting               |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               441 |                   256 |            697 |          139 |      8.15 |          15 |           34.98s |      2.89s |      37.87s | context-ignored                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               723 |                   491 |          1,214 |          167 |      15.1 |          34 |           37.17s |      7.38s |      44.56s | bullets(33)                       |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,653 |                   249 |          1,902 |         50.4 |      44.9 |          41 |           38.85s |      5.47s |      44.32s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |             1,653 |                   249 |          1,902 |         50.5 |      28.0 |          48 |           42.10s |      3.41s |      45.51s |                                   |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,871 |             1,305 |                   500 |          1,805 |          971 |      10.9 |          12 |           47.55s |      1.95s |      49.50s | lang_mixing, hallucination        |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     803 |            16,725 |                   500 |         17,225 |          367 |      70.6 |          13 |           53.07s |      1.82s |      54.89s | verbose                           |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |      12 |            16,723 |                   500 |         17,223 |          362 |      71.2 |          12 |           53.75s |      1.71s |      55.45s | verbose, bullets(55)              |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               442 |                   355 |            797 |          131 |      3.84 |          25 |           96.21s |      4.34s |     100.54s |                                   |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

## Model Gallery

Full output from each model:

<!-- markdownlint-disable MD013 MD033 -->

### ❌ microsoft/Florence-2-large-ft

**Status:** Failed (Weight Mismatch)
**Error:** Model loading failed: Missing 1 parameters:
language_model.lm_head.weight.
**Type:** `ValueError`
**Package:** `mlx`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5842, in _run_model_generation
    model, tokenizer, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5813, in _load_model
    model, tokenizer = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        lazy=params.lazy,
        ^^^^^^^^^^^^^^^^^
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 316, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 245, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 1 parameters:
language_model.lm_head.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5944, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5852, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: Missing 1 parameters:
language_model.lm_head.weight.
```

</details>

---

### ❌ mlx-community/InternVL3-14B-8bit

**Status:** Failed (Processor Error)
**Error:**

> Model loading failed: Received a InternVLImageProcessor for argument
> image_processor, but a ImageProcessingMixin was expected.
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5842, in _run_model_generation
    model, tokenizer, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5813, in _load_model
    model, tokenizer = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        lazy=params.lazy,
        ^^^^^^^^^^^^^^^^^
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 326, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 399, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/processing_auto.py", line 396, in from_pretrained
    return processor_class.from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/internvl_chat/processor.py", line 368, in from_pretrained
    return InternVLChatProcessor(
        image_processor=image_processor, tokenizer=tokenizer
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/internvl_chat/processor.py", line 288, in __init__
    super().__init__(image_processor, tokenizer, chat_template=chat_template)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 614, in __init__
    self.check_argument_for_proper_class(attribute_name, arg)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 697, in check_argument_for_proper_class
    raise TypeError(
        f"Received a {type(argument).__name__} for argument {argument_name}, but a {class_name} was expected."
    )
TypeError: Received a InternVLImageProcessor for argument image_processor, but a ImageProcessingMixin was expected.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5944, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5852, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: Received a InternVLImageProcessor for argument image_processor, but a ImageProcessingMixin was expected.
```

</details>

---

### ❌ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

**Status:** Failed (Lib Version)
**Error:**

> Model loading failed: cannot import name '_validate_images_text_input_order'
> from 'transformers.processing_utils'
> (/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py)
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5842, in _run_model_generation
    model, tokenizer, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5813, in _load_model
    model, tokenizer = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        lazy=params.lazy,
        ^^^^^^^^^^^^^^^^^
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 326, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 399, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/processing_auto.py", line 387, in from_pretrained
    processor_class = get_class_from_dynamic_module(
        processor_auto_map, pretrained_model_name_or_path, **kwargs
    )
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/dynamic_module_utils.py", line 583, in get_class_from_dynamic_module
    return get_class_in_module(class_name, final_module, force_reload=force_download)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/dynamic_module_utils.py", line 309, in get_class_in_module
    module_spec.loader.exec_module(module)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 1027, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/a04b2b044b1795d3e56eeee0d4946ca0c3a9d0fc/processing_kimi_vl.py", line 25, in <module>
    from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order
ImportError: cannot import name '_validate_images_text_input_order' from 'transformers.processing_utils' (/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5944, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5852, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: cannot import name '_validate_images_text_input_order' from 'transformers.processing_utils' (/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py)
```

</details>

---

### ❌ mlx-community/Kimi-VL-A3B-Thinking-8bit

**Status:** Failed (Lib Version)
**Error:**

> Model loading failed: cannot import name '_validate_images_text_input_order'
> from 'transformers.processing_utils'
> (/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py)
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5842, in _run_model_generation
    model, tokenizer, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5813, in _load_model
    model, tokenizer = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        lazy=params.lazy,
        ^^^^^^^^^^^^^^^^^
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 326, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 399, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/processing_auto.py", line 387, in from_pretrained
    processor_class = get_class_from_dynamic_module(
        processor_auto_map, pretrained_model_name_or_path, **kwargs
    )
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/dynamic_module_utils.py", line 583, in get_class_from_dynamic_module
    return get_class_in_module(class_name, final_module, force_reload=force_download)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/dynamic_module_utils.py", line 309, in get_class_in_module
    module_spec.loader.exec_module(module)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 1027, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/b5f89e575e95a48394b3d708beff29b842a4e14c/processing_kimi_vl.py", line 25, in <module>
    from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order
ImportError: cannot import name '_validate_images_text_input_order' from 'transformers.processing_utils' (/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5944, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5852, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: cannot import name '_validate_images_text_input_order' from 'transformers.processing_utils' (/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py)
```

</details>

---

### ❌ mlx-community/Phi-3.5-vision-instruct-bf16

**Status:** Failed (Config Missing)
**Error:**

> Model loading failed:
> /Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea
> does not appear to have a file named image_processing_phi3_v.py. Checkout
> 'https://huggingface.co//Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea/tree/main'
> for available files.
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5842, in _run_model_generation
    model, tokenizer, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5813, in _load_model
    model, tokenizer = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        lazy=params.lazy,
        ^^^^^^^^^^^^^^^^^
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 326, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 399, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/processing_auto.py", line 392, in from_pretrained
    return processor_class.from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1413, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1532, in _get_arguments_from_pretrained
    sub_processor = auto_processor_class.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, **kwargs
    )
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/image_processing_auto.py", line 610, in from_pretrained
    image_processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/dynamic_module_utils.py", line 572, in get_class_from_dynamic_module
    final_module = get_cached_module_file(
        repo_id,
    ...<7 lines>...
        repo_type=repo_type,
    )
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/dynamic_module_utils.py", line 390, in get_cached_module_file
    resolved_module_file = cached_file(
        pretrained_model_name_or_path,
    ...<8 lines>...
        _commit_hash=_commit_hash,
    )
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/utils/hub.py", line 276, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/utils/hub.py", line 377, in cached_files
    raise OSError(
    ...<2 lines>...
    )
OSError: /Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea does not appear to have a file named image_processing_phi3_v.py. Checkout 'https://huggingface.co//Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea/tree/main' for available files.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5944, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5852, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: /Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea does not appear to have a file named image_processing_phi3_v.py. Checkout 'https://huggingface.co//Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea/tree/main' for available files.
```

</details>

---

### ❌ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Status:** Failed (OOM)
**Error:**

> Model runtime error during generation for
> mlx-community/Qwen2-VL-2B-Instruct-4bit: [metal::malloc] Attempting to
> allocate 135383101952 bytes which is greater than the maximum allowed buffer
> size of 86586540032 bytes.
**Type:** `ValueError`
**Package:** `mlx`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5873, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        max_tokens=params.max_tokens,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 567, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 457, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 352, in generate_step
    mx.async_eval(y)
    ~~~~~~~~~~~~~^^^
RuntimeError: [metal::malloc] Attempting to allocate 135383101952 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5944, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5903, in _run_model_generation
    raise ValueError(msg) from gen_err
ValueError: Model runtime error during generation for mlx-community/Qwen2-VL-2B-Instruct-4bit: [metal::malloc] Attempting to allocate 135383101952 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

</details>

---

### ❌ mlx-community/deepseek-vl2-8bit

**Status:** Failed (Type Cast Error)
**Error:**

> Model runtime error during generation for mlx-community/deepseek-vl2-8bit:
> std::bad_cast
**Type:** `ValueError`
**Package:** `mlx`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5873, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        max_tokens=params.max_tokens,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 567, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 457, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 337, in generate_step
    outputs = model(input_ids, pixel_values, cache=prompt_cache, mask=mask, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/deepseek_vl_v2/deepseek_vl_v2.py", line 388, in __call__
    input_embeddings = self.get_input_embeddings(
        input_ids, pixel_values, images_spatial_crop, images_seq_mask
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/deepseek_vl_v2/deepseek_vl_v2.py", line 340, in get_input_embeddings
    total_tiles.append(pixel_values[idx, : batch_num_tiles[idx]])
                       ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: std::bad_cast

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5944, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5903, in _run_model_generation
    raise ValueError(msg) from gen_err
ValueError: Model runtime error during generation for mlx-community/deepseek-vl2-8bit: std::bad_cast
```

</details>

---

### ❌ prince-canuma/Florence-2-large-ft

**Status:** Failed (Model Error)
**Error:**

> Model loading failed: RobertaTokenizer has no attribute
> additional_special_tokens
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5842, in _run_model_generation
    model, tokenizer, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5813, in _load_model
    model, tokenizer = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        lazy=params.lazy,
        ^^^^^^^^^^^^^^^^^
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 326, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 399, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/processing_auto.py", line 392, in from_pretrained
    return processor_class.from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1414, in from_pretrained
    return cls.from_args_and_dict(args, processor_dict, **instantiation_kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1182, in from_args_and_dict
    processor = cls(*args, **valid_kwargs)
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/microsoft/Florence_hyphen_2_hyphen_large_hyphen_ft/4a12a2b54b7016a48a22037fbd62da90cd566f2a/processing_florence2.py", line 87, in __init__
    tokenizer.additional_special_tokens + \
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/tokenization_utils_base.py", line 1326, in __getattr__
    raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")
AttributeError: RobertaTokenizer has no attribute additional_special_tokens. Did you mean: 'add_special_tokens'?

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5944, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5852, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

</details>

---

### ✅ mlx-community/nanoLLaVA-1.5-4bit

**Metrics:** 386 TPS | 127 tokens

```text
The image is a vintage photograph of a historic town center in Wendover, England, captured during the early part of the year. The photograph is well-lit, with the sky suggesting it was taken during the late winter or early spring months. The scene is set against a backdrop of mature trees and moss-covered rooftops, indicating a traditional English market town. The image is described as a 'cool, quiet light' and a 'late winter afternoon' setting, with a focus on the historic buildings and the natural light. The image is cataloged under 'old house' and 'historic town center' in the context provided.
```

---

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

**Metrics:** 180 TPS | 84 tokens

```text
***

Historic Georgian Residence in Wendover, England
Description: A two-story brick house with white-framed windows and a white picket fence, surrounded by mature trees and a well-maintained lawn.
Keywords: Georgian architecture, brick house, white picket fence, mature trees, historic Wendover, England, classic sash windows, moss-covered roof tiles, traditional English market town.
```

---

### ✅ qnguyen3/nanoLLaVA

**Metrics:** 103 TPS | 143 tokens

```text
The image captures a serene winter afternoon in Wendover, England, with a picturesque old house as the focal point. The scene is set against the backdrop of a row of charming period cottages, and the old brick chimney adds a rustic charm to the setting. The image is rich in detail, with the old house's windows, the moss-covered clay tiles, and the green lawn all contributing to the overall atmosphere. The photo was taken on January 17, 2026, at 16:15:42 GMT, and was taken from a GPS 51.764250°N, 0.742750°W.
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 116 TPS | 123 tokens

```text
 The Old House, a Grade II* listed Georgian residence, is a prominent landmark in the historic town centre of Wendover, Buckinghamshire, England. The building, dating back to the early 18th century, stands alongside a row of charming period cottages, all of which are bathed in warm light as dusk falls. The bare, gnarled branches of mature trees frame the view, while the moss-covered clay tiles on the rooftops speak to the buildings' centuries-long history. The scene is captured from a low angle, emphasizing the grandeur of the Old House and its surroundings.
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 114 TPS | 125 tokens

```text
 The Old House is a Grade II* listed Georgian residence, dating back to the early 18th century. The building is characterized by its classic sash windows, which are a hallmark of Georgian architecture. The warm glow from the windows creates a welcoming atmosphere, contrasting with the crisp air of the winter evening. The bare, gnarled branches of the mature trees frame the view, while the moss-covered clay tiles on the rooftops add to the historical charm of the building. The scene is a typical example of a traditional English market town, with its neat white picket fence and verdant lawn.
```

---

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Metrics:** 287 TPS | 161 tokens

```text
Caption: "The Old House, Wendover, England - A timeless Georgian residence with a serene winter scene."

Description: The image showcases The Old House, a Grade II* listed Georgian residence in Wendover, Buckinghamshire, England. The house, dating back to the early 18th century, is characterized by its red brick construction, white-framed sash windows, and a prominent chimney. The scene is set in late winter, with bare trees and a moss-covered roof, creating a peaceful and historic atmosphere.

Keywords: Wendover, England, Georgian residence, The Old House, Grade II* listed, red brick, white-framed windows, moss-covered roof, late winter, peaceful atmosphere, historic town centre, traditional English market town.
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 114 TPS | 271 tokens

```text
 The Old House is a Grade II* listed Georgian residence, built in the early 18th century. The building is characterized by its symmetrical facade, with a central doorway flanked by two sash windows. The windows are framed with white trim, and the building is topped with a steeply pitched roof. The roof is covered in red tiles, and there are chimneys on either side of the building. The Old House is surrounded by a well-maintained lawn and a white picket fence.

To the right of the Old House, there is a row of charming period cottages. These cottages are also Georgian in style, with symmetrical facades and sash windows. The cottages are smaller than the Old House, and they have a more rustic appearance. The cottages are surrounded by a lawn and a white picket fence.

The town centre of Wendover is a charming place, with a mix of old and new buildings. The buildings are mostly made of brick, and they have a variety of architectural styles. The town centre is well-maintained, and it is a popular tourist destination.

The image captures the beauty of the town centre, and it is a reminder of the town's rich history. The Old House and the cottages are a testament to the town's enduring legacy.
```

---

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Metrics:** 114 TPS | 22 tokens

```text
Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 41.3 TPS | 59 tokens

```text
The photo was taken around 2026-01-17 16:15:42 GMT  at local time 16:15:42 . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.
```

---

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** 156 TPS | 342 tokens

```text
Given the provided context (Wendover, Buckinghamshire, England) and the visual cues from the image, here is a refined, factual caption and description along with appropriate keywords for cataloguing:

---

**Caption:**
A picturesque view of traditional English architecture in Wendover, showcasing a row of well-preserved 18th-century brick cottages. The central building features classic sash windows and a charming white picket fence, blending historical charm with serene rural beauty.

---

**Description:**
The image depicts a quaint row of brick buildings in Wendover, England, likely from the Georgian period. The central structure has a symmetrical facade with multiple sash windows, a small porch, and a chimney. The adjacent buildings share similar architectural features, including brickwork and white trim. The scene includes a well-maintained lawn, a narrow street, and mature trees with sparse foliage, suggesting a late autumn or winter setting. The buildings are likely residential, with a mix of historical and period-specific details.

---

**Keywords:**
- Georgian architecture, Wendover, England, UK
- Brick cottages, traditional English homes
- Sash windows, picket fences
- Historic town centre, rural England
- Late autumn/winter scene, Buckinghamshire
- Residential brick buildings, period style

---
**Note:** Since the exact date of the photo is not provided and the model is not capable of real-time location verification, I relied on the contextual clue of Wendover. For precise historical dates of the buildings, you would typically consult local historical records or architectural databases. The visual cues suggest a traditional English setting, but specific construction dates would require further research.
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 41.1 TPS | 321 tokens

```text
## Caption:

A picturesque view of The Old House, a Grade II* listed 18th-century building, stands proudly in the historic town centre of Wendover, Buckinghamshire, UK, as dusk settles. The charming brick facade with its symmetrical windows and white picket fence exudes a timeless appeal, framed by mature trees and a tranquil green lawn.

## Description:

This photograph captures the serene atmosphere of Wendover town centre at twilight. The focal point is The Old House, a beautifully preserved building dating back to the early 18th century and recognized for its Grade II* listed status. The building boasts a classic brick exterior, symmetrical rows of sash windows, and a welcoming dark green front door. A crisp white picket fence encloses a neat green lawn in front, adding to the idyllic scene.

Mature trees with gnarled branches stand on either side of the building, their silhouettes contrasting against the soft, fading light of the evening sky. The surrounding buildings, also constructed from brick, contribute to the historic character of the town. The road in the foreground is quiet, with a subtle curve leading into the scene. The overall impression is one of peaceful charm and historical significance, showcasing the quintessential appeal of a traditional English market town.

## Keywords:

Wendover, Buckinghamshire, England, UK, town centre, historic building, The Old House, Grade II* listed, 18th century, architecture, brick, sash windows, picket fence, trees, lawn, twilight, dusk, English countryside, traditional, peaceful, charming, heritage.
```

---

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Metrics:** 28.9 TPS | 248 tokens

```text
 The image depicts a quaint, historic town center in Wendover, Buckinghamshire, England. The focal point is The Old House, a Grade II* listed Georgian residence from the early 18th century. This building stands alongside a row of charming period cottages, all characterized by their classic sash windows and traditional architectural features.

The scene is bathed in the soft, cool light of a late winter afternoon, with the sun beginning to set. The warm glow from the windows contrasts with the crisp, cold air, creating a cozy ambiance. The bare, gnarled branches of mature trees frame the view, adding a touch of winter's chill to the scene. The moss-covered clay tiles on the rooftops indicate the buildings' long history, while the neat white picket fence and verdant lawn contribute to the timeless appeal of the traditional English market town.

The image captures the essence of a serene and picturesque town center, with its blend of historical architecture and natural elements. The Old House, with its Georgian design, is a notable feature, and the surrounding cottages add to the charm of the area. The overall atmosphere is one of tranquility and historical significance, reflecting the enduring beauty of Wendover's town center.<end_of_utterance>
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 42.1 TPS | 91 tokens

```text
The photo was taken around 2026-01-17 16:15:42 GMT  at local time 16:15:42  from GPS 51.764250°N, 0.742750°W . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.
```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 34.1 TPS | 349 tokens

```text
### Description
The image captures a serene winter afternoon in the historic town centre of Wendover, Buckinghamshire, England. The focal point is The Old House, a Grade II* listed Georgian residence built in the early 18th century. The house is characterized by its classic red brick facade, white-framed sash windows, and a moss-covered clay-tiled roof. The scene is framed by the bare branches of mature trees, adding to the tranquil atmosphere. A white picket fence and well-maintained lawn enhance the picturesque quality of the setting.

### Keywords
- Wendover, Buckinghamshire
- Georgian architecture
- Historic town centre
- The Old House
- Grade II* listed building
- Winter afternoon
- Sash windows
- Red brick
- Moss-covered roof
- Mature trees
- White picket fence
- Traditional English market town

### Cataloguing Information
- **Location:** Wendover, Buckinghamshire, England, United Kingdom
- **Date:** January 17, 2026
- **Time:** 16:15:42 GMT
- **Coordinates:** 51.764250°N, 0.742750°W
- **Subject:** The Old House, a Georgian residence
- **Season:** Winter
- **Weather:** Clear, late afternoon
- **Architectural Style:** Georgian
- **Listing Status:** Grade II* listed building
- **Surroundings:** Period cottages, mature trees, well-maintained lawn, white picket fence

This detailed description and set of keywords should facilitate effective cataloguing and searching for the image.
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 27.5 TPS | 387 tokens

```text
## Image Catalogue Information:

**Caption:** A row of brick buildings, including the Grade II* listed “The Old House”, along the town centre street in Wendover, Buckinghamshire, England. The image was captured in late winter (January) at dusk, with warm light emanating from the windows.

**Description:** A wide-angle, ground-level photograph depicting a row of predominantly red brick buildings in Wendover town centre. The largest building, identified as “The Old House”, is a two-story structure with a symmetrical facade, multiple sash windows, and a central doorway. To the right of The Old House are smaller, adjoining brick cottages. All buildings feature pitched, moss-covered clay tile roofs. A neat white picket fence separates the buildings from a well-maintained lawn. Bare trees frame the view. The sky is a muted twilight blue. A paved street is visible in the foreground. The image conveys a quiet, historic atmosphere.

**Keywords:**

*   Wendover
*   Buckinghamshire
*   England
*   United Kingdom
*   UK
*   Town Centre
*   Historic Buildings
*   Georgian Architecture
*   The Old House
*   Grade II* Listed Building
*   Cottages
*   Brick Buildings
*   Sash Windows
*   Clay Tiles
*   Winter
*   January
*   Dusk
*   Twilight
*   Street Scene
*   Architecture
*   Exterior
*   Trees
*   Picket Fence
*   51.764250°N, 0.742750°W
*   2026-01-17
*   16:15:42 GMT
*   Traditional English Town
*   Moss Covered Roofs
*   Red Brick
*   Exterior Architecture




```

---

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** 28.1 TPS | 245 tokens

```text
Caption:ĊHistoricĠGeorgianĠarchitectureĠinĠtheĠheartĠofĠWendover,ĠBuckinghamshire,ĠEngland.ĠTheĠOldĠHouse,ĠaĠGradeĠII*ĠlistedĠbuildingĠfromĠtheĠearlyĠ18thĠcentury,ĠstandsĠalongsideĠcharmingĠperiodĠcottagesĠunderĠaĠlateĠwinterĠafternoonĠsky.ĊĊDescription:ĊTheĠimageĠshowcasesĠaĠrowĠofĠtraditionalĠEnglishĠbuildingsĠinĠWendoverĠtownĠcentre.ĠTheĠfocalĠpointĠisĠTheĠOldĠHouse,ĠaĠtwo-storeyĠGeorgianĠstructureĠwithĠredĠbrickĠwallsĠandĠwhite-framedĠsashĠwindows.ĠTheĠbuildingĠfeaturesĠaĠcentralĠdoorĠwithĠaĠsmallĠporchĠandĠaĠlargeĠgarageĠdoorĠtoĠtheĠright.ĠAĠwhiteĠpicketĠfenceĠrunsĠalongĠtheĠfront,ĠwithĠaĠwell-maintainedĠlawnĠandĠmatureĠtreesĠframingĠtheĠscene.ĠTheĠsurroundingĠcottagesĠshareĠsimilarĠarchitecturalĠstyles,ĠwithĠmoss-coveredĠclayĠtileĠroofsĠandĠchimneyĠstacks.ĠTheĠskyĠisĠclearĠwithĠaĠsoft,ĠcoolĠlightĠtypicalĠofĠaĠlateĠwinterĠafternoon.ĊĊKeywords:ĊWendover,ĠBuckinghamshire,ĠEngland,ĠUK,ĠhistoricĠarchitecture,ĠGeorgianĠhouse,ĠGradeĠII*ĠlistedĠbuilding,ĠperiodĠcottages,ĠredĠbrick,ĠsashĠwindows,ĠpicketĠfence,Ġlawn,ĠmatureĠtrees,ĠwinterĠafternoon,ĠtownĠcentre,ĠtraditionalĠEnglishĠbuildings,Ġmoss-coveredĠroofs,ĠchimneyĠstacks,ĠlateĠwinterĠlight.
```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 35.9 TPS | 500 tokens

```text
Here are my reasoning steps:
The user asks: "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image. Context: The image relates to ', Town centre, Wendover, England, United Kingdom, UK". Then they provide a sample caption that is quite elaborate, but they also say "The photo was taken around 2026-01-17 16:15:42 GMT at local time 16:15:42 from GPS 51.764250°N, 0.742750°W . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate."

Thus we need to produce a factual caption, description, and keywords. The image is a photo of a historic brick house with white-framed windows, a white picket fence, a large tree, a green lawn, a driveway, and adjacent brick cottages. The location is Wendover town centre, England, UK. The building appears to be a Georgian style house, possibly Grade II* listed. The time appears to be late winter/early spring, maybe January, with bare trees. The lighting suggests early evening or late afternoon. The sky is clear or partly cloudy. The house has a tall chimney. The roof is tiled. The house has a black front door and a black garage door. The adjacent houses have similar brickwork but different colors. The road is a quiet street.

We need to produce a factual caption, description, and keywords. The caption should be concise but descriptive, focusing on visual content. The description should be more detailed, perhaps a paragraph. The keywords should be a list of terms for cataloguing/search.

We must not speculate beyond what we can see. The user gave a sample caption that includes "The Old House, a distinctive Grade II* listed Georgian residence dating to the early 18th century". That is speculation unless we know that. The user says "Do not speculate." So we should not assume the building is Grade II* listed or name it "The Old House". We can describe it as a "two-story brick house with white-framed sash windows, black front door, black garage door, white picket fence, mature trees, green lawn, adjacent brick cottages, quiet street, Wendover town centre". We can mention the architectural style
```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 19.1 TPS | 391 tokens

```text
### Caption
The cool, quiet light of a late winter afternoon settles over the historic town centre of Wendover, Buckinghamshire, England. This January scene captures The Old House, a distinctive Grade II* listed Georgian residence dating to the early 18th century, standing alongside a row of charming period cottages. As dusk begins to fall, warm light glows from the classic sash windows, offering a welcoming contrast to the crisp air. The bare, gnarled branches of mature trees frame the view, while the moss-covered clay tiles on the rooftops speak to the buildings' centuries-long history. The scene, with its neat white picket fence and verdant lawn, encapsulates the timeless appeal of a traditional English market town.

### Description
The image depicts a picturesque street scene in Wendover, Buckinghamshire, England. The focal point is The Old House, a well-preserved Georgian residence from the early 18th century, which is a Grade II* listed building. The house features classic sash windows and a moss-covered clay tile roof, indicative of its historical significance. The surrounding area includes a row of period cottages, all contributing to the quaint, historic charm of the town. The scene is framed by mature trees with bare branches, and the lawn in front of The Old House is neatly maintained with a white picket fence. The image captures the serene and timeless appeal of a traditional English market town during the late winter afternoon.

### Keywords
- Wendover, Buckinghamshire
- The Old House
- Georgian residence
- Grade II* listed building
- Historic town centre
- Period cottages
- Sash windows
- Moss-covered clay tile roof
- Mature trees
- White picket fence
- Traditional English market town
- Late winter afternoon
- Historic architecture
- Timeless appeal
- Serene scene
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 17.5 TPS | 500 tokens

```text
A front view of a large, red brick house with a white picket fence and a garage on the right side of the house. The garage is dark green and has a white door. The house has a large number of windows with white frames. The windows are lined up vertically and have white trim. The garage is on the right side of the house, and the garage door is dark green. The garage is dark green and has a white door. The garage is on the right side of the house. The garage is dark green and has a white door. The garage is on the right side of the house. The house has a large number of windows with white trim. The windows are lined up vertically and have white trim. The garage is on the right side of the house. The garage door is dark green and has a white door. The garage is on the right side of the house. The house has a large number of windows with white trim. The windows are lined up vertically and have white trim. The garage is on the right side of the house. The garage door is dark green and has a white door. The garage is on the right side of the house. The house has a large number of windows with white trim. The windows are lined up vertically and have white trim. The garage is on the right side of the house. The garage door is dark green and has a white door. The garage is on the right side of the house. The house has a large number of windows with white trim. The windows are lined up vertically and have white trim. The garage is on the right side of the house. The garage door is dark green and has a white door. The garage is on the right side of the house. The house has a large number of windows with white trim. The windows are lined up vertically and have white trim. The garage is on the right side of the house. The garage door is dark green and has a white door. The garage is on the right side of the house. The house has a large number of windows with white trim. The windows are lined up vertically and have white trim. The garage is on the right side of the house. The garage door is dark green and has a white door. The garage is on the right side of the house. The house has a large number of windows with white trim. The windows are lined up vertically and have white trim. The garage is on the right side of the house. The
```

---

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

**Metrics:** 64.2 TPS | 500 tokens

```text
<think>Got it, let's tackle this. The user wants a factual caption, description, and keywords for cataloging or searching.

First, I need to extract key details from the provided context. The image is of the town centre of Wendover, England. The main subject is "The Old House," a Grade II* listed Georgian residence from the early 18th century. It's part of a row of period cottages.

The time and date are given: 2026-01-17 16:15:42 GMT, local time 16:15:42. GPS coordinates are 51.764250°N, 0.742750°W.

Now, structure the output: a factual caption, a detailed description, and relevant keywords.

For the caption, it should be concise but capture the key elements: location, main subject, time of day, atmosphere.

The description should elaborate on the visual content, including architectural details (brick, sash windows, moss-covered roofs), landscape elements (trees, picket fence, lawn), lighting (late winter afternoon, dusk), and historical context (Grade II* listed, early 18th century).

Keywords should be a list of terms that would help in cataloging or searching. These should include location terms (Wendover, Buckinghamshire, England, UK), architectural terms (Georgian, brick, sash windows, picket fence, moss-covered roof, Grade II* listed), time and weather terms (late winter afternoon, dusk, cool light, crisp air), landscape terms (trees, lawn, white picket fence, moss-covered clay tiles).

Now, put it all together in the required format.</think>
<|begin_of_box|>### Factual Caption  
The historic town centre of Wendover, Buckinghamshire, England, is captured on a late winter afternoon, with the cool, quiet light of dusk beginning to fall. The scene features “The Old House,” a Grade II* listed Georgian residence dating to the early 18th century, standing alongside a row of charming period cottages.  

### Detailed Description  
The image depicts a serene, historic streetscape in Wendover, England. The focal point is “The Old House,” a two-story brick building with white sash windows, a moss-covered clay tile roof, and a prominent stone chimney. The entrance, marked by the number “34,” features a dark green door flanked by wall-mounted lanterns
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 4.92 TPS | 140 tokens

```text
The photo is oriented in a three-quarter view, facing the bottom left corner of the frame. The foreground shows a dark gray asphalt road with a double yellow line in the center. To the left of the road is a green grass lawn with a white picket fence and two large, leafless trees. The trees have a dark brown, gnarled trunk with a dark green door on the left and a dark green garage door on the right. The house has a red brick exterior with white window frames and a dark green door. The roof is made of dark gray clay tiles with a dark green chimney on the left side. The sky is a light blue, with a few thin clouds visible.
```

---

### ✅ mlx-community/GLM-4.6V-Flash-6bit

**Metrics:** 46.4 TPS | 500 tokens

```text
<think>Got it, let's tackle this. The user wants a factual caption, description, and keywords for cataloguing/searching.

First, I need to extract the key elements from the image and context.

The image shows a historic building in Wendover, UK. The main building is "The Old House," a Georgian residence. There are other period cottages next to it.

Visual elements: brick construction, white-framed sash windows, moss-covered clay roof tiles, chimney stacks with red pots, white picket fence, mature trees (bare branches, gnarled trunks), grassy lawn, paved driveway, road with double yellow lines.

Contextual info: location (Wendover, Buckinghamshire, England, UK), date/time (2026-01-17 16:15:42 GMT, local time 16:15:42), GPS coordinates (51.764250°N, 0.742750°W).

Now, structure the caption, description, and keywords.

Caption should be concise but include key info: location, main subject, date/time context.

Description should elaborate on visual elements, materials, architectural style, surrounding landscape, and contextual details (date, time, GPS).

Keywords should be specific terms for cataloguing/searching, including location, building type, architectural style, materials, time of day, season, and any notable features (chimneys, picket fence, sash windows).

Let me draft each part.

Caption: Historic Georgian residence "The Old House" in Wendover town centre, Buckinghamshire, England, captured on 17 January 2026 at 16:15 local time.

Description: The image depicts "The Old House," a Grade II* listed early 18th-century Georgian residence, situated in the historic town centre of Wendover, Buckinghamshire, England. The two-story brick building features white-framed sash windows, a moss-covered clay tile roof, and a prominent chimney stack with red pots. Adjacent to "The Old House" is a row of charming period cottages, also constructed from brick with white-framed windows and moss-covered roofs. The scene is framed by mature trees with bare, gnarled branches, and a neatly maintained lawn bordered by a white picket fence. A paved driveway leads to a large dark green garage door integrated into the main building. The image was captured on 17 January 202
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 8.15 TPS | 256 tokens

```text
I'm happy to help you with your question. However, I want to clarify that I'm a large language model, I don't have the ability to see or interact with the physical world, so I won't be able to provide a visual description of the image.

That being said, I can try to help you with your question. Based on your description of the image, it seems like you're looking for a way to describe the scene in a way that's engaging and easy to visualize. Here are a few suggestions:

* Use vivid and descriptive language to paint a picture of the scene. For example, you could describe the house as "a charming, old-fashioned home with a warm and welcoming atmosphere."
* Use sensory details to bring the scene to life. For example, you could describe the sound of the wind rustling through the trees, the smell of the flowers, or the feel of the sun on your skin.
* Use metaphors or similes to add depth and interest to your description. For example, you could describe the house as "a cozy little nest" or "a warm and welcoming hug."

I hope these suggestions are helpful. Let me know if you have any other questions or if there's anything else I can help you with.
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 15.1 TPS | 491 tokens

```text
Okay, here's a factual caption, description, and keywords suitable for cataloguing/searching for the image, based on your provided information and visual analysis. I've aimed for specificity and avoided speculation.

**Caption:**

The Old House, a Grade II* listed Georgian residence in Wendover, Buckinghamshire, England, photographed on January 17, 2026, at 16:15 GMT. The image depicts the brick facade of the house and adjacent cottages, illuminated by late afternoon light.

**Description:**

A photograph showing the front elevation of The Old House, a substantial two-story brick building with a symmetrical arrangement of windows. The house is flanked by smaller brick cottages. A white picket fence separates the buildings from a lawn. Mature trees are positioned in front of the house, partially obscuring the facade. The scene is lit by the soft light of late afternoon, with some windows showing interior illumination. The image was taken in Wendover town centre, Buckinghamshire, England. The road is visible in the foreground. Moss is visible on the clay tile roofs.

**Keywords:**

*   Wendover
*   Buckinghamshire
*   England
*   United Kingdom
*   UK
*   Town Centre
*   The Old House
*   Georgian Architecture
*   Grade II* Listed Building
*   Historic Building
*   Cottages
*   Brick Architecture
*   January
*   Winter
*   Afternoon Light
*   Trees
*   Picket Fence
*   Clay Tiles
*   Moss
*   2026
*   51.764250°N, 0.742750°W
*   Exterior
*   Architecture
*   Buildings
*   Residential
*   Landscape
*   Traditional Architecture
*   18th Century Architecture
*   Sash Windows



**Notes:**

*   I've included the GPS coordinates as a keyword for precise location searching.
*   I've focused on observable details (materials, architectural style, time of year) rather than making interpretations about the "feel" of the scene.
*   The date and time of the photograph are included as keywords for chronological searching.
*   I've used a mix of broad and specific keywords to maximize searchability.
```

---

### ✅ mlx-community/Molmo-7B-D-0924-8bit

**Metrics:** 44.9 TPS | 249 tokens

```text
 Here is a caption for the image, crafted according to your specifications:

***

The cool, quiet light of a late winter afternoon settles over the historic town centre of Wendover, Buckinghamshire, England. This January scene captures The Old House, a distinctive Grade II* listed Georgian residence dating to the early 18th century, standing alongside a row of charming period cottages.

As dusk begins to fall, warm light glows from the classic sash windows, offering a welcoming contrast to the crisp air. The bare, gnarled branches of mature trees frame the view, while the moss-covered clay tiles on the rooftops speak to the buildings' centuries-long history. The scene, with its neat white picket fence and verdant lawn, encapsulates the timeless appeal of a traditional English market town.'

The photo was taken around 2026-01-17 16:15:42 GMT  at local time 16:15:42  from GPS 51.764250°N, 0.742750°W . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.
```

---

### ✅ mlx-community/Molmo-7B-D-0924-bf16

**Metrics:** 28 TPS | 249 tokens

```text
 Here is a caption for the image, crafted according to your specifications:

***

The cool, quiet light of a late winter afternoon settles over the historic town centre of Wendover, Buckinghamshire, England. This January scene captures The Old House, a distinctive Grade II* listed Georgian residence dating to the early 18th century, standing alongside a row of charming period cottages.

As dusk begins to fall, warm light glows from the classic sash windows, offering a welcoming contrast to the crisp air. The bare, gnarled branches of mature trees frame the view, while the moss-covered clay tiles on the rooftops speak to the buildings' centuries-long history. The scene, with its neat white picket fence and verdant lawn, encapsulates the timeless appeal of a traditional English market town.'

The photo was taken around 2026-01-17 16:15:42 GMT  at local time 16:15:42  from GPS 51.764250°N, 0.742750°W . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.
```

---

### ✅ microsoft/Phi-3.5-vision-instruct

**Metrics:** 10.9 TPS | 500 tokens

```text
The Old House, a Grade II* listed Georgian building, stands prominently in the town centre of Wendover, Buckinghamshire, England. Dating back to the early 18th century, this historic residence is a notable feature of the area, surrounded by period cottages and a white picket fence. The image captures the building during a late winter afternoon, with the warm glow of sash windows illuminating the scene. The bare trees and moss-covered clay tiles on the rooftops add to the building's historical charm, reflecting its long-standing presence in the town. The serene atmosphere and traditional architecture evoke the timeless appeal of a traditional English market town.<|end|><|endoftext|>


Here is a chart:

| Year | Number of Students |
|------|--------------------|
| 2010 | 120                |
| 2011 | 130                |
| 2012 | 140                |
| 2013 | 150                |
| 2014 | 160                |
| 2015 | 170                |
| 2016 | 180                |
| 2017 | 190                |
| 2018 | 200                |
| 2019 | 210                |
| 2020 | 220                |
| 2021 | 230                |
| 2022 | 240                |
| 2023 | 250                |
| 2024 | 260                |
| 2025 | 270                |
| 2026 | 280                |
| 2027 | 290                |
| 2028 | 300                |
| 2029 | 310                |
| 2030 | 320                |
| 2031 | 330                |
| 2032 |
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 70.6 TPS | 500 tokens

```text
Okay, let's tackle this request. The user provided a detailed image description and a caption, and they want a factual caption, description, and keywords for cataloging or searching. The context is the town centre of Wendover, England, UK.

First, I need to make sure I understand the task. They want me to create three things: a factual caption, a description, and keywords. The caption should be factual, based on the image, not the provided caption. Wait, no—the user says "Here is a caption for the image, crafted according to your specifications." But then they say "Do not speculate. Focus on visual content..." So the caption they provided is an example, but I need to create a new one based on the image.

Wait, the user says: "Here is a caption for the image, crafted according to your specifications: [the example]". Then they say "The photo was taken around 2026-01-17 16:15:42 GMT...". Wait, the date is in the future? But the context is Wendover, England. Maybe it's a typo, but the user says "Do not speculate." So I should stick to the image as is.

Wait, the user provided a caption that's supposed to be the answer, but they want me to create a factual caption, description, and keywords. Wait, no—the user says: "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image."

Wait, the context is Wendover, England. The image shows a historic house, The Old House, which is Grade II* listed. The caption they provided is an example, but they want me to create a new one based on the image.

Wait, the user says: "Here is a caption for the image, crafted according to your specifications: [the example]". So the example is the caption they want me to follow, but they want me to create a factual caption, description, and keywords. Wait, maybe the example is the correct one, but I need to make sure it's factual.

Wait, the user says: "Do not speculate. Focus on visual content, drawing on any available contextual information for specificity. Do not speculate."

So I need to create a caption that's factual, based on the image. The example they provided is a caption, but maybe they want me to adjust it to be more
```

---

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 71.2 TPS | 500 tokens

```text
### Caption
A serene, late afternoon view of The Old House in Wendover, Buckinghamshire, England. The historic brick building, with its distinctive white sash windows and a prominent chimney, stands amidst a quiet, tree-lined street. The scene is bathed in soft, cool light, highlighting the moss-covered roof and the well-maintained green lawn. A white picket fence encloses the property, adding to the picturesque charm of this traditional English town.

### Description
The image captures a tranquil scene in the town centre of Wendover, Buckinghamshire, England. The focal point is The Old House, a Grade II* listed Georgian residence built in the early 18th century. The house features a red brick facade, white sash windows, and a prominent chimney. The roof is covered in moss, indicating its age and historical significance. A large, mature tree with bare branches stands in the foreground, framing the view. The property is enclosed by a white picket fence, and a small, well-maintained lawn adds to the charm. The surrounding area includes other period cottages, and the scene is set against a backdrop of a clear sky with a few clouds. The overall atmosphere is peaceful and timeless, reflecting the character of a traditional English market town.

### Keywords
- Wendover
- Buckinghamshire
- England
- UK
- The Old House
- Grade II*
- Georgian architecture
- Historic building
- Brick house
- Sash windows
- Chimney
- Moss-covered roof
- White picket fence
- Tree
- Late winter
- Traditional English town
- Historical architecture
- Grade II* listed building
- Period cottages
- Rural England
- English countryside
- Historic village
- Residential property
- Town centre
- Late afternoon
- Cool light
- Soft lighting
- Timeless appeal
- Traditional architecture
- English market town
- Historic house
- Brick and stone
- White window frames
- Green lawn
- Bare tree branches
- Cloudy sky
- Moss
- Charming village
- Rural setting
- English countryside
- Historic house
- Period building
- Traditional English house
- Grade II* listed
- Historic building
- English countryside
- Traditional architecture
- Rural England
- Historic village
- Residential property
- Town centre
- Late afternoon
- Cool light
- Soft lighting
- Timeless appeal
-
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.84 TPS | 355 tokens

```text
The image depicts a large, old house with a red brick exterior and white trim, featuring multiple windows and a chimney. The house is situated on a street with a sidewalk and a tree in front of it. The image appears to be a photograph of a historic building in a small town or village.

* A large, old house with a red brick exterior and white trim:
	+ The house has a classic, traditional design with a symmetrical facade.
	+ The red brick exterior is weathered and has a warm, earthy tone.
	+ The white trim adds a touch of elegance and sophistication to the house.
* A tree in front of the house:
	+ The tree is tall and has a large trunk.
	+ The tree has a few branches and leaves, but it is not fully grown.
	+ The tree is situated on the left side of the house, creating a sense of balance and harmony.
* A sidewalk and a street in front of the house:
	+ The sidewalk is made of concrete and is well-maintained.
	+ The street is paved and has a yellow line down the middle.
	+ The street is empty, with no cars or people in sight.
* A chimney on the roof of the house:
	+ The chimney is tall and has a distinctive shape.
	+ The chimney is made of brick and has a warm, earthy tone.
	+ The chimney is situated on the right side of the house, adding a touch of character to the building.

Overall, the image presents a serene and peaceful scene, with the old house and its surroundings evoking a sense of history and tradition. The image may be used to showcase the beauty of traditional architecture or to highlight the importance of preserving historical buildings.
```

---

<!-- markdownlint-enable MD013 MD033 -->

---

## System/Hardware Information

- **OS**: Darwin 25.2.0
- **macOS Version**: 26.2
- **SDK Version**: 26.2
- **Python Version**: 3.13.9
- **Architecture**: arm64
- **GPU/Chip**: Apple M4 Max
- **GPU Cores**: 40
- **Metal Support**: Metal 4
- **RAM**: 128.0 GB
- **CPU Cores (Physical)**: 16
- **CPU Cores (Logical)**: 16

## Library Versions

- `Pillow`: `12.1.0`
- `huggingface-hub`: `1.3.3`
- `mlx`: `0.30.4.dev20260123+dc81c150`
- `mlx-lm`: `0.30.5`
- `mlx-vlm`: `0.3.10`
- `numpy`: `2.4.1`
- `tokenizers`: `0.22.2`
- `transformers`: `5.0.0rc3`

_Report generated on: 2026-01-23 21:53:39 GMT_
