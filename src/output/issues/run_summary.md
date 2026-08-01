# mlx-vlm compatibility findings across 62 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-01 22:52:46 BST
- *Evaluation mode:* blind
- *Models attempted:* 62
- *Completed:* 60
- *Crashed:* 1
- *Indeterminate:* 1
- *Crashes requiring action:* 1
- *Other results requiring review:* 27

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

Parameterized reproduction

```bash
python reproduce.py mlx-community/Step-3.7-Flash-oQ2e --revision 3dacb46f724ac89725bcd922fb779c7ed1499fe7 --image cats.jpg --prompt-file prompt.txt
```

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-step-37-flash-oq2e) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Step-3.7-Flash-oQ2e.md) |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/FastVLM-0.5B-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-fastvlm-05b-bf16) |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-v-46-8bit) |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | Missing or empty fields: Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-nanollava-15-4bit) |
| qnguyen3/nanoLLaVA | unusable | Missing or empty fields: Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-qnguyen3-nanollava) |
| mlx-community/LFM2.5-VL-1.6B-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-lfm25-vl-16b-bf16) |
| HuggingFaceTB/SmolVLM-Instruct | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-huggingfacetb-smolvlm-instruct) |
| mlx-community/SmolVLM-Instruct-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-smolvlm-instruct-bf16) |
| mlx-community/GLM-4.6V-Flash-6bit | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-6bit) |
| mlx-community/Idefics3-8B-Llama3-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | unusable | Missing or empty fields: Title, Keywords; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16) |
| mlx-community/gemma-3n-E2B-4bit | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3n-e2b-4bit) |
| mlx-community/paligemma2-10b-ft-docci-448-6bit | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-6bit) |
| mlx-community/gemma-4-31b-bf16 | unusable | No response text was returned; Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-4-31b-bf16) |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit | unusable | Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit) |
| mlx-community/paligemma2-3b-ft-docci-448-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-ft-docci-448-bf16) |
| mlx-community/GLM-4.6V-nvfp4 | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Kimi-VL-A3B-Thinking-8bit | unusable | Missing or empty fields: Title; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Conversation-role control tokens remain visible; Internal reasoning text remains visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit) |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llava-v16-mistral-7b-8bit) |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | Missing or empty fields: Title, Description; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/paligemma2-10b-ft-docci-448-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-bf16) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | Response repeats the same text; Missing or empty fields: Title; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning text remains visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |

## Indeterminate attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/Qwen3.5-35B-A3B-bf16 | not evaluated | none | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen35-35b-a3b-bf16) |

## Clean completions

34 clean completions; see the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* cats.jpg
- *Generation: max_tokens:* 500
- *Generation: prefill_step_size:* 4096
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *mlx-vlm:* 0.6.8
- *mlx:* 0.32.1.dev20260801+fb5133e10
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
