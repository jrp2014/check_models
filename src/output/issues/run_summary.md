# mlx-vlm compatibility findings across 62 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-02 01:13:51 BST
- *Evaluation mode:* assisted
- *Models attempted:* 62
- *Completed:* 60
- *Crashed:* 1
- *Indeterminate:* 1
- *Crashes requiring action:* 1
- *Other results requiring review:* 31

Observations are mechanical facts from one image, not general model-quality
judgements.

## Crashes requiring action

### mlx-community/Step-3.7-Flash-oQ2e

- *Execution / usability:* crashed / not evaluated
- *Phase:* processor_load
- *Stage:* Processor Error
- *Resolved revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7

Root exception chain

```text
ValueError: Loaded processor has no image_processor; expected multimodal processor.
caused by: ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,240 pixels
- *Image size:* 61,337,614 bytes
- *Image SHA-256:* a907e72a592bcdbdc026c71ac8a508d6cf87ee27222afaf7cc26f96145151f89

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-01 16:28:40 UTC+01:00
- GPS: 52.345200°N, 1.503700°E

Descriptive hints:
- Title hint: Town centre, Halesworth, England, UK, GBR, Europe
- Description hint: The Cut in Halesworth, Suffolk in the UK
- Keyword hints: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:
```

</details>

The original local input is not published, so this report does not claim a
complete reproduction command. Use a shareable equivalent image or add the
original image before filing.

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-step-37-flash-oq2e) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Step-3.7-Flash-oQ2e.md) |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | Missing or empty fields: Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-nanollava-15-4bit) |
| HuggingFaceTB/SmolVLM-Instruct | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-huggingfacetb-smolvlm-instruct) |
| mlx-community/SmolVLM-Instruct-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-smolvlm-instruct-bf16) |
| qnguyen3/nanoLLaVA | unusable | Missing or empty fields: Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-qnguyen3-nanollava) |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | Missing or empty fields: Title, Description; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-v-46-8bit) |
| mlx-community/FastVLM-0.5B-bf16 | unusable | Missing or empty fields: Title, Description; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-fastvlm-05b-bf16) |
| mlx-community/paligemma2-10b-ft-docci-448-6bit | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-6bit) |
| mlx-community/LFM2-VL-1.6B-8bit | unusable | Response repeats the same text; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-lfm2-vl-16b-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | Missing or empty fields: Title, Description; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3n-e4b-it-bf16) |
| mlx-community/paligemma2-10b-ft-docci-448-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-bf16) |
| mlx-community/gemma-3n-E2B-4bit | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3n-e2b-4bit) |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llava-v16-mistral-7b-8bit) |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit | unusable | Missing or empty fields: Title; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit) |
| mlx-community/gemma-4-31b-bf16 | unusable | No response text was returned; Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-4-31b-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-8bit | unusable | Missing or empty fields: Title, Description; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit) |
| mlx-community/paligemma2-3b-ft-docci-448-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-ft-docci-448-bf16) |
| mlx-community/GLM-4.6V-Flash-6bit | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-6bit) |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | Missing or empty fields: Keywords; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | Response repeats the same text; Unrecognised model control tokens remain visible; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/Molmo-7B-D-0924-bf16 | unusable | No response text was returned; Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-molmo-7b-d-0924-bf16) |
| mlx-community/Molmo-7B-D-0924-8bit | unusable | No response text was returned; Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-molmo-7b-d-0924-8bit) |
| mlx-community/MolmoPoint-8B-fp16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-molmopoint-8b-fp16) |
| mlx-community/GLM-4.6V-nvfp4 | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16) |
| mlx-community/X-Reasoner-7B-8bit | unusable | Response repeats the same text; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | unusable | Missing or empty fields: Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | Missing or empty fields: Title, Description; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |

## Indeterminate attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | not evaluated | none | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx) |

## Clean completions

30 clean completions; see the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,240 pixels, 61,337,614 bytes
- *Generation: max_tokens:* 500
- *Generation: prefill_step_size:* 4096
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *mlx-vlm:* 0.6.8
- *mlx:* 0.32.1.dev20260802+fb5133e10
- *transformers:* 5.14.1
- *macOS Version:* 26.6
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.13.13

## Full artifacts

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) |
| Model gallery | [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) |
| Run JSON | [run.json](https://github.com/jrp2014/check_models/blob/main/src/output/run.json) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
