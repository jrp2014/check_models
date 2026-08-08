# mlx-vlm compatibility findings across 64 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-08 13:18:49 BST
- *Evaluation mode:* assisted
- *Models attempted:* 64
- *Completed:* 63
- *Crashed:* 1
- *Indeterminate:* 0
- *Crashes requiring action:* 1
- *Other results requiring review:* 44

Observations are mechanical facts from one image, not general model-quality
judgements.

## Crashes requiring action

### mlx-community/Inkling-Small-mlx-4bit

- *Execution / usability:* crashed / not evaluated
- *Phase:* model_load
- *Stage:* Model Error
- *Resolved revision:* f0cafad5b1a3e54be06ba03fe07b4cd4e8bcc612

Root exception chain

```text
ValueError: Received 362 parameters not in model; families: audio_tower, language_model; representative parameters: audio_tower.encoder.biases, audio_tower.encoder.scales, language_model.model.layers.10.mlp.experts.down_proj.biases.
caused by: ValueError: Model loading failed: Received 362 parameters not in model; families: audio_tower, language_model; representative parameters: audio_tower.encoder.biases, audio_tower.encoder.scales, language_model.model.layers.10.mlp.experts.down_proj.biases.
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,964 x 5,605 pixels
- *Image size:* 39,212,214 bytes
- *Image SHA-256:* f5cc97b21d6d751921d8c5b18cbc80b9b8bca1839b8ff95e1a75d7427992e488

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-07 17:17:16 UTC+01:00

Descriptive hints:
- Title hint: Seafront, Seaford, England, UK, GBR, Europe
- Description hint: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.
- Keyword hints: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful

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
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-inkling-small-mlx-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/issues/issue_mlx-community_Inkling-Small-mlx-4bit.md) |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Title or keywords do not meet requested constraints | 15 |
| Required fields are missing or empty | 8 |
| Response repeats the same text; Required fields are missing or empty; Response appears cut off at the token limit | 3 |
| Required fields are missing or empty; Extra text appears before the Title field | 3 |
| Required fields are missing or empty; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 3 |
| Title or keywords do not meet requested constraints; Title, Description and Keywords copy all supplied hints unchanged | 3 |
| Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field | 2 |
| No response text was returned; Required fields are missing or empty | 1 |
| Response repeats the same text; Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 1 |
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 1 |
| Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | 1 |
| Required fields are missing or empty; Response appears cut off at the token limit | 1 |
| Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 1 |
| Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/gemma-4-31b-bf16 | unusable | No response text was returned; Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-4-31b-bf16) |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit | unusable | Response repeats the same text; Missing or empty fields: Title; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/gemma-3n-E2B-4bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3n-e2b-4bit) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/X-Reasoner-7B-8bit | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 6 words (requested 5-10); Keyword list has 166 terms (requested 10-18); Duplicate keywords: seaford, england, uk, europe, inflatable boat, motorboat, people, horizon, blue sky, clear, water, action, holiday, white wake, driver, yamaha, rigid hull, lady maverick, yamaha engine, clear day, open sea, summer, adventure, travel, tourism, seaside, gbr, blue, horizon line, inflatable, outboard motor, speed boat | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/GLM-4.6V-Flash-6bit | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-6bit) |
| mlx-community/GLM-4.6V-nvfp4 | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | Missing or empty fields: Title, Description, Keywords; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | unusable | Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | unusable | Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/FastVLM-0.5B-bf16 | unusable | Missing or empty fields: Title, Description; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-fastvlm-05b-bf16) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | Missing or empty fields: Title, Description; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-8bit | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit) |
| mlx-community/LFM2.5-VL-1.6B-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-lfm25-vl-16b-bf16) |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-llava-v16-mistral-7b-8bit) |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-v-46-8bit) |
| mlx-community/MolmoPoint-8B-fp16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-molmopoint-8b-fp16) |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | Missing or empty fields: Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-nanollava-15-4bit) |
| mlx-community/paligemma2-10b-ft-docci-448-6bit | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-6bit) |
| mlx-community/paligemma2-10b-ft-docci-448-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-bf16) |
| mlx-community/paligemma2-3b-ft-docci-448-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-ft-docci-448-bf16) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16) |
| qnguyen3/nanoLLaVA | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-qnguyen3-nanollava) |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |
| HuggingFaceTB/SmolVLM-Instruct | usable with caveats | Title has 6 words (requested 5-10); Keyword list has 20 terms (requested 10-18); Title, Description and Keywords copy all supplied hints unchanged | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-huggingfacetb-smolvlm-instruct) |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable with caveats | Title has 6 words (requested 5-10); Keyword list has 18 terms (requested 10-18); Duplicate keywords: seaford, uk | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-liquidai-lfm25-vl-450m-mlx-bf16) |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable with caveats | Title has 7 words (requested 5-10); Keyword list has 24 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-devstral-small-2-24b-instruct-2512-5bit) |
| mlx-community/gemma-3-27b-it-qat-4bit | usable with caveats | Title has 8 words (requested 5-10); Keyword list has 24 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3-27b-it-qat-4bit) |
| mlx-community/gemma-3n-E4B-it-bf16 | usable with caveats | Title has 7 words (requested 5-10); Keyword list has 19 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3n-e4b-it-bf16) |
| mlx-community/GLM-4.6V-Flash-mxfp4 | usable with caveats | Title has 3 words (requested 5-10); Keyword list has 13 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-mxfp4) |
| mlx-community/LFM2-VL-1.6B-8bit | usable with caveats | Title has 2 words (requested 5-10); Keyword list has 14 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-lfm2-vl-16b-8bit) |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | Title has 7 words (requested 5-10); Keyword list has 20 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-ministral-3-14b-instruct-2512-nvfp4) |
| mlx-community/Molmo-7B-D-0924-8bit | usable with caveats | Title has 5 words (requested 5-10); Keyword list has 19 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-molmo-7b-d-0924-8bit) |
| mlx-community/Molmo-7B-D-0924-bf16 | usable with caveats | Title has 5 words (requested 5-10); Keyword list has 19 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-molmo-7b-d-0924-bf16) |
| mlx-community/Ornith-1.0-35B-bf16 | usable with caveats | Title has 8 words (requested 5-10); Keyword list has 19 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-ornith-10-35b-bf16) |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | usable with caveats | Title has 6 words (requested 5-10); Keyword list has 20 terms (requested 10-18); Title, Description and Keywords copy all supplied hints unchanged | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit) |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | usable with caveats | Title has 5 words (requested 5-10); Keyword list has 18 terms (requested 10-18); Duplicate keywords: action, vehicles | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16) |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable with caveats | Title has 10 words (requested 5-10); Keyword list has 20 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen35-9b-mlx-4bit) |
| mlx-community/Qwen3.6-27B-mxfp8 | usable with caveats | Title has 6 words (requested 5-10); Keyword list has 19 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen36-27b-mxfp8) |
| mlx-community/SmolVLM-Instruct-bf16 | usable with caveats | Title has 6 words (requested 5-10); Keyword list has 20 terms (requested 10-18); Title, Description and Keywords copy all supplied hints unchanged | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-smolvlm-instruct-bf16) |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable with caveats | Title has 2 words (requested 5-10); Keyword list has 20 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx) |
| Qwen/Qwen3-VL-2B-Instruct | usable with caveats | Title has 5 words (requested 5-10); Keyword list has 18 terms (requested 10-18); Duplicate keywords: action, vehicles | [diagnostics](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md#diagnostic-qwen-qwen3-vl-2b-instruct) |

## Clean completions

19 clean completions; see the full model gallery (model_gallery.md, producer-local).

## Run context

- *Image:* JPEG, 9,964 x 5,605 pixels, 39,212,214 bytes
- *Generation: max_tokens:* 500
- *Generation: prefill_step_size:* 4096
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.8.9
- *check_models revision:* 32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.11
- *mlx:* 0.32.1.dev20260808+6539d1807
- *transformers:* 5.14.1
- *macOS Version:* 26.6
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.13.13

GitHub links are pinned to producer commit `32d71ddc969b`, so the linked
evidence is durable.

## Full artifacts

Stale retained artifacts omitted because their timestamps fall outside this
run: `check_models.log`, `environment.log`.

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/reports/diagnostics.md) |
| Model gallery | model_gallery.md (producer-local, not published) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/results.jsonl) |
| Run JSON | [run.json](https://github.com/jrp2014/check_models/blob/32d71ddc969b3cbe8dd8c0ffb64ad6b00e419901/src/output/run.json) |
