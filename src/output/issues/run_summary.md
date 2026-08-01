# mlx-vlm compatibility findings across 62 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-01 11:44:29 BST
- *Evaluation mode:* blind
- *Models attempted:* 62
- *Completed:* 61
- *Crashed:* 1
- *Indeterminate:* 0
- *Actionable failures:* 1
- *Other surfaced results:* 28

Observations are mechanical facts from one image, not general model-quality
judgements.

## Actionable failures

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

## Other surfaced results

| Model | Execution / usability | Observations | Full evidence |
| --- | --- | --- | --- |
| mlx-community/nanoLLaVA-1.5-4bit | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-nanollava-15-4bit) |
| mlx-community/MiniCPM-V-4.6-8bit | completed / unusable | missing requested sections, unexpected catalog preamble | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-v-46-8bit) |
| mlx-community/FastVLM-0.5B-bf16 | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-fastvlm-05b-bf16) |
| qnguyen3/nanoLLaVA | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-qnguyen3-nanollava) |
| mlx-community/LFM2.5-VL-1.6B-bf16 | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-lfm25-vl-16b-bf16) |
| HuggingFaceTB/SmolVLM-Instruct | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-huggingfacetb-smolvlm-instruct) |
| mlx-community/SmolVLM-Instruct-bf16 | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-smolvlm-instruct-bf16) |
| mlx-community/GLM-4.6V-Flash-6bit | completed / unusable | missing requested sections, unexpected catalog preamble, unexpected special token | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-6bit) |
| mlx-community/Idefics3-8B-Llama3-bf16 | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | completed / unusable | missing requested sections, unexpected catalog preamble, unexpected special token | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/gemma-4-31b-bf16 | completed / unusable | empty output, missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-4-31b-bf16) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | completed / unusable | missing requested sections, unexpected catalog preamble, unexpected special token | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/pixtral-12b-8bit | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-pixtral-12b-8bit) |
| mlx-community/GLM-4.6V-nvfp4 | completed / unusable | missing requested sections, unexpected catalog preamble, unexpected special token | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | completed / unusable | missing requested sections, token cap truncation, prompt instruction echo, unexpected catalog preamble | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16) |
| mlx-community/gemma-3n-E2B-4bit | completed / unusable | missing requested sections, token cap truncation | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3n-e2b-4bit) |
| mlx-community/pixtral-12b-bf16 | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-pixtral-12b-bf16) |
| mlx-community/paligemma2-10b-ft-docci-448-6bit | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-6bit) |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit | completed / unusable | token cap truncation, unexpected catalog preamble, thinking trace present, thinking trace incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit) |
| mlx-community/Kimi-VL-A3B-Thinking-8bit | completed / unusable | missing requested sections, token cap truncation, prompt instruction echo, unexpected catalog preamble, thinking trace present, role boundary token present | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit) |
| mlx-community/paligemma2-3b-ft-docci-448-bf16 | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-ft-docci-448-bf16) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | completed / unusable | missing requested sections, token cap truncation | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/llava-v1.6-mistral-7b-8bit | completed / unusable | repeated output, missing requested sections, token cap truncation | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llava-v16-mistral-7b-8bit) |
| mlx-community/paligemma2-3b-pt-896-4bit | completed / unusable | repeated output, missing requested sections, token cap truncation, prompt instruction echo, unexpected catalog preamble | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | completed / unusable | missing requested sections, token cap truncation, prompt instruction echo, unexpected catalog preamble | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | completed / unusable | repeated output, token cap truncation, unexpected catalog preamble | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/paligemma2-10b-ft-docci-448-bf16 | completed / unusable | missing requested sections | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | completed / unusable | repeated output, missing requested sections, token cap truncation, unexpected catalog preamble, thinking trace present | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |

## Clean completions

33 clean completions; see the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* cats.jpg
- *Generation: max_tokens:* 500
- *Generation: prefill_step_size:* 4096
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *mlx-vlm:* 0.6.8
- *mlx:* 0.32.1.dev20260731+fb5133e10
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
