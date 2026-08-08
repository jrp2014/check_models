# mlx-vlm compatibility findings across 64 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-08 12:40:35 BST
- *Evaluation mode:* blind
- *Models attempted:* 64
- *Completed:* 63
- *Crashed:* 1
- *Indeterminate:* 0
- *Crashes requiring action:* 1
- *Other results requiring review:* 10

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
- *Image dimensions:* 640 x 480 pixels
- *Image size:* 173,131 bytes
- *Image SHA-256:* dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e

<details>
<summary>Exact prompt</summary>

```text
Describe this image
```

</details>

The original local input is not published, so this report does not claim a
complete reproduction command. Use a shareable equivalent image or add the
original image before filing.

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md#diagnostic-mlx-community-inkling-small-mlx-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/issues/issue_mlx-community_Inkling-Small-mlx-4bit.md) |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Conversation-role control tokens remain visible | 3 |
| Unrecognised model control tokens remain visible | 2 |
| Response appears cut off at the token limit | 2 |
| Response repeats the same text; Response appears cut off at the token limit | 1 |
| Response appears cut off at the token limit; Internal reasoning block appears incomplete | 1 |
| Response is unusually short | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/gemma-3n-E2B-4bit | unusable | Response repeats the same text; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3n-e2b-4bit) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md#diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-8bit | unusable | Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit) |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md#diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit) |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |
| mlx-community/paligemma2-3b-pt-896-4bit | usable with caveats | Response is unusually short | [diagnostics](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |

## Clean completions

53 clean completions; see the full model gallery (model_gallery.md, producer-local).

## Run context

- *Image:* JPEG, 640 x 480 pixels, 173,131 bytes
- *Generation: max_tokens:* 500
- *Generation: prefill_step_size:* 4096
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.8.9
- *check_models revision:* 00ce6a08c5df718aeb04738406e08a81f28a7304
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.11
- *mlx:* 0.32.1.dev20260808+6539d1807
- *transformers:* 5.14.1
- *macOS Version:* 26.6
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.13.13

GitHub links are pinned to producer commit `00ce6a08c5df`, so the linked
evidence is durable.

## Full artifacts

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/reports/diagnostics.md) |
| Model gallery | model_gallery.md (producer-local, not published) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/results.jsonl) |
| Run JSON | [run.json](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/run.json) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/00ce6a08c5df718aeb04738406e08a81f28a7304/src/output/environment.log) |
| Log | check_models.log (producer-local, not published) |
