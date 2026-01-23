# Model Performance Results

_Generated on 2026-01-23 14:11:58 GMT_

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (`No Chat Template`)

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `mlx-vlm` | 1 | No Chat Template | `mlx-community/gemma-3n-E2B-4bit` |

### Actionable Items by Package

#### mlx-vlm

- **mlx-community/gemma-3n-E2B-4bit** (No Chat Template)
  - Error: `Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! Fo...`
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

**Overall runtime:** 4.89s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                        |   Generation (s) |   Load (s) |   Total (s) | Quality Issues   |   Error Package |
|:----------------------------------|-----------------:|-----------:|------------:|:-----------------|----------------:|
| `mlx-community/gemma-3n-E2B-4bit` |                  |            |             |                  |         mlx-vlm |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

## Model Gallery

Full output from each model:

<!-- markdownlint-disable MD013 MD033 -->

### ❌ mlx-community/gemma-3n-E2B-4bit

**Status:** Failed (No Chat Template)
**Error:**

> Cannot use chat template functions because tokenizer.chat_template is not
> set and no template argument was passed! For information about writing
> templates and setting the tokenizer.chat_template attribute, please see the
> documentation at
> https://huggingface.co/docs/transformers/main/en/chat_templating
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5944, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 5856, in _run_model_generation
    formatted_prompt: str | list[Any] = apply_chat_template(
                                        ~~~~~~~~~~~~~~~~~~~^
        processor=tokenizer,
        ^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        num_images=1,
        ^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 501, in apply_chat_template
    return get_chat_template(processor, messages, add_generation_prompt)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 386, in get_chat_template
    return processor.apply_chat_template(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        messages,
        ^^^^^^^^^
    ...<2 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/tokenization_utils_base.py", line 3111, in apply_chat_template
    chat_template = self.get_chat_template(chat_template, tools)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/tokenization_utils_base.py", line 3293, in get_chat_template
    raise ValueError(
    ...<4 lines>...
    )
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating
```

</details>

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
- `mlx`: `0.30.4.dev20260123+1650c490`
- `mlx-lm`: `0.30.5`
- `mlx-vlm`: `0.3.10`
- `numpy`: `2.4.1`
- `tokenizers`: `0.22.2`
- `transformers`: `5.0.0rc3`

_Report generated on: 2026-01-23 14:11:58 GMT_
