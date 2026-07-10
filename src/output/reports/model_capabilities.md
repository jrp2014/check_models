# Model Capability Scorecard

Generated on: 2026-07-10 21:31:49 BST

- Mode: triage
- Grounding: not grounded; caption hygiene only
- Coverage: current run plus 100 prior history run(s).

Structured metadata and keyword capability: not evaluated in triage mode.
Use the Clean, Hygiene, Caption, speed, and memory columns as a caption-review shortlist; visual correctness still requires grounded metadata or manual review.

<!-- markdownlint-disable MD034 -->

| Model                                                   | Use                 | Current    |   Runs | Success   | Clean   | Hygiene   | Caption   | Gen TPS   | Peak GB   | Latest signal                         |
|---------------------------------------------------------|---------------------|------------|--------|-----------|---------|-----------|-----------|-----------|-----------|---------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | caption             | passed     |     60 | 100%      | 85%     | 100       | 95        | 102       | 17        | clean                                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     | caption             | passed     |      7 | 100%      | 100%    | 100       | 93        | 60.7      | 71        | clean                                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | caption             | passed     |     26 | 96%       | 88%     | 100       | 93        | 267       | 3.0       | clean                                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        | caption             | passed     |     94 | 98%       | 22%     | 100       | 88        | 127       | 5.5       | clean                                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | caption             | passed     |     92 | 97%       | 35%     | 100       | 96        | 20.2      | 15        | clean                                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | caption             | passed     |     92 | 97%       | 24%     | 100       | 96        | 91.9      | 7.7       | clean                                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | caption             | passed     |     92 | 97%       | 35%     | 100       | 92        | 60.4      | 9.2       | clean                                 |
| `mlx-community/InternVL3-14B-8bit`                      | caption             | passed     |     92 | 97%       | 37%     | 100       | 91        | 33.1      | 19        | clean                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | caption             | passed     |     92 | 97%       | 53%     | 100       | 90        | 63.9      | 10        | clean                                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   | caption             | passed     |     92 | 97%       | 23%     | 100       | 88        | 127       | 5.5       | clean                                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | caption             | passed     |     92 | 97%       | 36%     | 100       | 86        | 51.3      | 20        | clean                                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | caption             | passed     |     92 | 97%       | 23%     | 100       | 84        | 63.3      | 10        | clean                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | caption             | passed     |     92 | 97%       | 54%     | 100       | 84        | 66.3      | 9.7       | clean                                 |
| `mlx-community/GLM-4.6V-nvfp4`                          | caption             | passed     |     92 | 97%       | 23%     | 100       | 80        | 48.6      | 63        | clean                                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | caption             | passed     |     92 | 97%       | 35%     | 100       | 77        | 4.9       | 25        | clean                                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | caption             | passed     |     92 | 97%       | 52%     | 100       | 74        | 15.9      | 32        | clean                                 |
| `qnguyen3/nanoLLaVA`                                    | caption             | passed     |     92 | 96%       | 39%     | 100       | 96        | 114       | 4.0       | clean                                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | caption             | passed     |     92 | 96%       | 33%     | 100       | 88        | 61.9      | 9.7       | clean                                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | caption             | passed     |     92 | 97%       | 35%     | 80        | 93        | 33.7      | 19        | formatting                            |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | caption             | passed     |     92 | 97%       | 46%     | 95        | 74        | 195       | 4.5       | clean                                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | needs-review        | passed     |     92 | 97%       | 41%     | 100       | 66        | 28.9      | 18        | clean                                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | needs-review        | passed     |     57 | 100%      | 49%     | 67        | 60        | 24.3      | 30        | clean                                 |
| `mlx-community/gemma-4-31b-it-4bit`                     | caption             | passed     |     76 | 95%       | 60%     | 100       | 96        | 25.8      | 20        | clean                                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         | caption             | passed     |     17 | 82%       | 76%     | 100       | 92        | 22.5      | 28        | clean                                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | caption             | passed     |     92 | 96%       | 39%     | 100       | 77        | 349       | 1.9       | clean                                 |
| `mlx-community/InternVL3-8B-bf16`                       | caption             | passed     |     92 | 95%       | 46%     | 100       | 89        | 34.0      | 18        | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | caption             | passed     |     82 | 94%       | 28%     | 100       | 88        | 116       | 22        | clean                                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | caption             | passed     |     92 | 95%       | 36%     | 100       | 87        | 30.3      | 27        | clean                                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          | caption             | passed     |     21 | 81%       | 76%     | 100       | 94        | 25.6      | 29        | clean                                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | needs-review        | passed     |     92 | 97%       | 32%     | 85        | 62        | 46.2      | 17        | clean                                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | caption             | passed     |     82 | 96%       | 38%     | 81        | 84        | 96.4      | 7.3       | clean                                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | caption             | passed     |     92 | 92%       | 35%     | 100       | 96        | 17.7      | 30        | clean                                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               | needs-review        | passed     |     22 | 100%      | 59%     | 60        | 56        | 139       | 5.2       | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | caption             | passed     |     92 | 92%       | 24%     | 100       | 87        | 98.5      | 30        | clean                                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | needs-review        | passed     |     92 | 96%       | 34%     | 85        | 62        | 130       | 5.5       | hit token cap (200)                   |
| `mlx-community/gemma-4-31b-bf16`                        | current-run-blocked | failed     |     75 | 92%       | 25%     | 88        | 96        | 7.5       | 64        | mlx-vlm; model-error                  |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | caption             | passed     |     92 | 92%       | 23%     | 92        | 89        | 307       | 2.5       | clean                                 |
| `mlx-community/Qwen3.5-27B-4bit`                        | caption             | passed     |     92 | 92%       | 23%     | 100       | 79        | 32.3      | 19        | clean                                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | caption             | passed     |     92 | 91%       | 36%     | 100       | 90        | 323       | 2.1       | clean                                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | needs-review        | passed     |     92 | 97%       | 35%     | 50        | 62        | 77.2      | 16        | reasoning-leak                        |
| `microsoft/Phi-3.5-vision-instruct`                     | caption-review      | passed     |     92 | 97%       | 5%      | 56        | 77        | 57.9      | 9.2       | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | avoid               | passed     |     92 | 94%       | 29%     | 67        | 70        | 71.0      | 71        | clean                                 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | avoid               | passed     |     45 | 73%       | 51%     | 100       | 69        | 544       | 1.0       | clean                                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | avoid               | passed     |     92 | 92%       | 33%     | 67        | 68        | 73.7      | 10        | clean                                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | avoid               | passed     |     92 | 88%       | 37%     | 85        | 62        | 19.4      | 10        | clean                                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | avoid               | passed     |     92 | 88%       | 35%     | 85        | 62        | 5.3       | 26        | clean                                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | avoid               | passed     |     92 | 88%       | 35%     | 85        | 62        | 33.2      | 11        | clean                                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | avoid               | passed     |     92 | 83%       | 35%     | 100       | 69        | 315       | 3.0       | clean                                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             | avoid               | passed     |     57 | 88%       | 35%     | 60        | 56        | 126       | 5.1       | clean                                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | caption-review      | passed     |     92 | 96%       | 3%      | 38        | 75        | 31.9      | 19        | clean                                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | caption-review      | passed     |     92 | 52%       | 15%     | 100       | 88        | 193       | 4.1       | clean                                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | avoid               | caveat     |     92 | 88%       | 0%      | 15        | 29        | 77.8      | 4.6       | harness:long-context; context-budget  |
| `mlx-community/MolmoPoint-8B-fp16`                      | unstable            | passed     |     91 | 20%       | 13%     | 85        | 62        | 5.9       | 23        | clean                                 |
| `mlx-community/pixtral-12b-bf16`                        | avoid               | avoid      |     92 | 97%       | 14%     | 0         | 62        | 19.7      | 27        | cutoff                                |
| `mlx-community/pixtral-12b-8bit`                        | avoid               | avoid      |     92 | 97%       | 13%     | 0         | 62        | 38.8      | 15        | cutoff                                |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | avoid               | avoid      |     92 | 97%       | 9%      | 0         | 62        | 4.5       | 39        | cutoff; reasoning-leak                |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | avoid               | avoid      |     92 | 97%       | 0%      | 0         | 54        | 42.3      | 14        | cutoff; reasoning-leak                |
| `mlx-community/gemma-3n-E2B-4bit`                       | avoid               | avoid      |     92 | 97%       | 0%      | 0         | 39        | 116       | 6.0       | repetitive; cutoff                    |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | avoid               | avoid      |     92 | 89%       | 0%      | 0         | 54        | 63.9      | 60        | cutoff; degeneration                  |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | avoid               | avoid      |     56 | 88%       | 0%      | 0         | 68        | 132       | 5.2       | harness:stop-token                    |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | avoid               | avoid      |     92 | 63%       | 2%      | 0         | 62        | 70.1      | 20        | cutoff; reasoning-leak                |
| `prince-canuma/Florence-2-large-ft`                     | unstable            | not-tested |     11 | 27%       | 0%      | -         | -         | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL         |
| `mlx-community/granite-4.1-8b-mxfp8`                    | integration-blocked | not-tested |      1 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                       |
| `mlx-community/deepseek-vl2-8bit`                       | integration-blocked | not-tested |     10 | 0%        | 0%      | -         | -         | -         | -         | MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR |
| `microsoft/Florence-2-large-ft`                         | integration-blocked | not-tested |     11 | 0%        | 0%      | -         | -         | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL         |
| `ggml-org/gemma-3-1b-it-GGUF`                           | integration-blocked | not-tested |     11 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                       |
| `facebook/pe-av-large`                                  | integration-blocked | not-tested |     21 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                       |
<!-- markdownlint-enable MD034 -->

Scores aggregate available current/history signals. Older history rows that
predate capability fields still count toward reliability, but not caption,
keyword, or metadata averages.
