# mlx-vlm compatibility findings across 42 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-09 00:14:53 BST
- *Evaluation mode:* assisted
- *Models attempted:* 42
- *Completed:* 41
- *Crashed:* 1
- *Indeterminate:* 0
- *Crashes requiring action:* 1
- *Other results requiring review:* 16

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
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 60,712,161 bytes
- *Image SHA-256:* 2d3e8ab39253f25bfa3f4a37188a72d369bb79657c8f7011611e1f58fb3afc23

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-08 15:43:55 UTC+01:00
- GPS: 51.815915°N, 0.638706°W

Descriptive hints:
- Title hint: Town centre, Tring, England, UK, GBR, Europe
- Description hint: Akeman Street Baptist Church, Tring, Herts
- Keyword hints: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky

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
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-inkling-small-mlx-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Inkling-Small-mlx-4bit.md) |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 1 |
| Response repeats the same text; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field | 1 |
| Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | 1 |
| Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Expected model wrapper tokens remain visible; Title or keywords do not meet requested constraints | 1 |
| Required fields are missing or empty; Extra text appears before the Title field; Conversation-role control tokens remain visible | 1 |
| Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Expected model wrapper tokens remain visible | 1 |
| Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 1 |
| Extra text appears before the Title field; Expected model wrapper tokens remain visible | 4 |
| Extra text appears before the Title field; Expected model wrapper tokens remain visible; Title or keywords do not meet requested constraints | 2 |
| Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 2 |
| Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | unusable | Response repeats the same text; Response appears cut off at the token limit; Keyword list has 469 terms (requested 10-18); Duplicate keywords: church | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-liquidai-lfm25-vl-450m-mlx-bf16) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | unusable | Response repeats the same text; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Expected model wrapper tokens remain visible; Keyword list has 223 terms (requested 10-18); Duplicate keywords: tring, hertfordshire, england, uk, europe, red brick, steeple, stone wall, gate, entrance, sign, sky, clouds, bushes, herts, baptist church, church | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | Missing or empty fields: Title; Extra text appears before the Title field; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |
| mlx-community/gemma-4-26b-a4b-it-4bit | unusable | Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Keyword list has 36 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-4-26b-a4b-it-4bit) |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Expected model wrapper tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-v-46-8bit) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Keyword list has 39 terms (requested 10-18); Duplicate keywords: hertfordshire, red brick, chapel, entrance, clouds, church, england, sign, bushes | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| mlx-community/GLM-4.6V-nvfp4 | unusable | Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title has 15 words (requested 5-10); Keyword list has 66 terms (requested 10-18); Duplicate keywords: hertfordshire, england, europe, red brick, sign, bushes, chapel, church, clouds, entrance, maroon door, arched windows, greenery | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Ornith-1.0-35B-bf16 | unusable | Extra text appears before the Title field; Expected model wrapper tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ornith-10-35b-bf16) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | unusable | Extra text appears before the Title field; Expected model wrapper tokens remain visible; Keyword list has 19 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16) |
| mlx-community/Qwen3.5-35B-A3B-4bit | unusable | Extra text appears before the Title field; Expected model wrapper tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen35-35b-a3b-4bit) |
| mlx-community/Qwen3.5-9B-MLX-4bit | unusable | Extra text appears before the Title field; Expected model wrapper tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen35-9b-mlx-4bit) |
| mlx-community/Qwen3.6-27B-mxfp8 | unusable | Extra text appears before the Title field; Expected model wrapper tokens remain visible; Keyword list has 20 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen36-27b-mxfp8) |
| mlx-community/Step-3.7-Flash-oQ2e | unusable | Extra text appears before the Title field; Expected model wrapper tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-step-37-flash-oq2e) |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |

## Clean completions

7 clean completions; 18 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 60,712,161 bytes
- *Generation: enable_thinking:* true
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: thinking_budget:* 300
- *Generation: thinking_end_token:* "&lt;/think&gt;"
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.9.0
- *check_models revision:* 96883994f5f5a716ec67b0b8d73d6f3a12e7748d
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.11
- *mlx:* 0.32.1.dev20260808+8d6662986
- *transformers:* 5.14.1
- *macOS Version:* 26.6
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.13.13

GitHub links target the repository's mutable main branch; they resolve to this
run's evidence only once these artifacts are committed, and a later run's
commit supersedes them. Pin links to that artifact commit when durable issue
evidence is required.

## Full artifacts

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) |
| Model gallery | [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) |
| Run JSON | [run.json](https://github.com/jrp2014/check_models/blob/main/src/output/run.json) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
