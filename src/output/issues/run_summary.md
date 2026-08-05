# mlx-vlm compatibility findings across 63 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-02 22:46:09 BST
- *Evaluation mode:* blind
- *Models attempted:* 63
- *Completed:* 63
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 35

Observations are mechanical facts from one image, not general model-quality
judgements.

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/gemma-4-31b-bf16 | unusable | No response text was returned; Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-4-31b-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | Response repeats the same text; Missing or empty fields: Title; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning text remains visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Title has 6 words (requested 5-10); Keyword list has 147 terms (requested 10-18); Duplicate keywords: household, household frames | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llava-v16-mistral-7b-8bit) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/GLM-4.6V-Flash-6bit | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-6bit) |
| mlx-community/GLM-4.6V-nvfp4 | unusable | Unrecognised model control tokens remain visible; Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| HuggingFaceTB/SmolVLM-Instruct | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-huggingfacetb-smolvlm-instruct) |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | Missing or empty fields: Title, Description; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/FastVLM-0.5B-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-fastvlm-05b-bf16) |
| mlx-community/gemma-3n-E2B-4bit | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3n-e2b-4bit) |
| mlx-community/Idefics3-8B-Llama3-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-8bit | unusable | Missing or empty fields: Title; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Conversation-role control tokens remain visible; Internal reasoning text remains visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit) |
| mlx-community/LFM2.5-VL-1.6B-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-lfm25-vl-16b-bf16) |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | Missing or empty fields: Title; Extra text appears before the Title field | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-v-46-8bit) |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | Missing or empty fields: Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-nanollava-15-4bit) |
| mlx-community/paligemma2-10b-ft-docci-448-6bit | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-6bit) |
| mlx-community/paligemma2-10b-ft-docci-448-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-bf16) |
| mlx-community/paligemma2-3b-ft-docci-448-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-ft-docci-448-bf16) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | unusable | Missing or empty fields: Title, Keywords; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16) |
| mlx-community/SmolVLM-Instruct-bf16 | unusable | Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-smolvlm-instruct-bf16) |
| qnguyen3/nanoLLaVA | unusable | Missing or empty fields: Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-qnguyen3-nanollava) |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit | unusable | Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible; Title has 10 words (requested 5-10); Keyword list has 16 terms (requested 10-18); Duplicate keywords: pink fabric | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible; Title has 21 words (requested 5-10); Keyword list has 13 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable with caveats | Title has 5 words (requested 5-10); Keyword list has 5 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-liquidai-lfm25-vl-450m-mlx-bf16) |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable with caveats | Title has 7 words (requested 5-10); Keyword list has 19 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-devstral-small-2-24b-instruct-2512-5bit) |
| mlx-community/gemma-3n-E4B-it-bf16 | usable with caveats | Title has 3 words (requested 5-10); Keyword list has 14 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3n-e4b-it-bf16) |
| mlx-community/GLM-4.6V-Flash-mxfp4 | usable with caveats | Title has 5 words (requested 5-10); Keyword list has 4 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-mxfp4) |
| mlx-community/LFM2-VL-1.6B-8bit | usable with caveats | Title has 5 words (requested 5-10); Keyword list has 11 terms (requested 10-18); Duplicate keywords: cats | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-lfm2-vl-16b-8bit) |
| mlx-community/Ornith-1.0-35B-bf16 | usable with caveats | Title has 9 words (requested 5-10); Keyword list has 20 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ornith-10-35b-bf16) |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | usable with caveats | Title has 7 words (requested 5-10); Keyword list has 4 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit) |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable with caveats | Title has 11 words (requested 5-10); Keyword list has 3 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx) |

## Clean completions

28 clean completions; see the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 640 x 480 pixels, 173,131 bytes
- *Generation: max_tokens:* 500
- *Generation: prefill_step_size:* 4096
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.8.9
- *check_models revision:* 3599a4baa107c687bc992190808e5e4545f69b6b
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.8
- *mlx:* 0.32.1.dev20260802+fb5133e10
- *transformers:* 5.14.1
- *macOS Version:* 26.6
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.13.13

GitHub links target the repository's mutable main branch; use the committed
output snapshot when durable issue evidence is required.

## Full artifacts

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) |
| Model gallery | [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) |
| Run JSON | [run.json](https://github.com/jrp2014/check_models/blob/main/src/output/run.json) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
