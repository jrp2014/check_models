# Model Capability Scorecard

Generated on: 2026-07-03 13:59:09 BST

- Mode: triage
- Grounding: not grounded; caption hygiene only
- Coverage: current run plus 100 prior history run(s).

Structured metadata and keyword capability: not evaluated in triage mode.
Use the Clean, Hygiene, Caption, speed, and memory columns as a caption-review shortlist; visual correctness still requires grounded metadata or manual review.

<!-- markdownlint-disable MD034 -->

| Model                                                   | Use                 |   Runs | Success   | Clean   | Hygiene   | Caption   | Gen TPS   | Peak GB   | Latest signal                         |
|---------------------------------------------------------|---------------------|--------|-----------|---------|-----------|-----------|-----------|-----------|---------------------------------------|
| `mlx-community/Ornith-1.0-35B-bf16`                     | caption             |      1 | 100%      | 100%    | 100       | 96        | 63.4      | 71        | clean                                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | caption             |     54 | 100%      | 83%     | 100       | 95        | 112       | 17        | clean                                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | caption             |     20 | 100%      | 90%     | 100       | 93        | 265       | 3.0       | clean                                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | caption             |     51 | 100%      | 49%     | 100       | 89        | 19.2      | 30        | clean                                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               | needs-review        |     16 | 100%      | 62%     | 85        | 62        | 130       | 5.2       | clean                                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         | caption             |     11 | 91%       | 82%     | 67        | 93        | 26.1      | 28        | model-config; processor-error         |
| `HuggingFaceTB/SmolVLM-Instruct`                        | caption             |     94 | 98%       | 16%     | 100       | 88        | 129       | 5.5       | clean                                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | caption             |     92 | 97%       | 28%     | 100       | 96        | 18.2      | 15        | clean                                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | caption             |     92 | 97%       | 17%     | 100       | 96        | 93.2      | 7.7       | clean                                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | caption             |     92 | 97%       | 28%     | 100       | 92        | 61.3      | 9.2       | clean                                 |
| `mlx-community/InternVL3-14B-8bit`                      | caption             |     92 | 97%       | 30%     | 100       | 91        | 32.9      | 19        | clean                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | caption             |     92 | 97%       | 47%     | 100       | 89        | 59.8      | 10        | clean                                 |
| `mlx-community/gemma-4-31b-it-4bit`                     | caption             |     70 | 96%       | 59%     | 100       | 96        | 24.1      | 19        | clean                                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          | caption             |     15 | 87%       | 80%     | 67        | 94        | 20.6      | 29        | model-config; processor-error         |
| `mlx-community/SmolVLM-Instruct-bf16`                   | caption             |     92 | 97%       | 16%     | 100       | 88        | 128       | 5.5       | clean                                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | caption             |     92 | 97%       | 29%     | 100       | 86        | 51.0      | 20        | clean                                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | caption             |     92 | 97%       | 16%     | 100       | 83        | 63.7      | 10        | clean                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | caption             |     92 | 97%       | 48%     | 100       | 82        | 60.0      | 9.7       | clean                                 |
| `mlx-community/GLM-4.6V-nvfp4`                          | caption             |     92 | 97%       | 16%     | 100       | 82        | 45.5      | 63        | clean                                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | caption             |     76 | 96%       | 36%     | 100       | 80        | 99.2      | 7.0       | clean                                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | caption             |     92 | 97%       | 28%     | 100       | 77        | 5.1       | 25        | clean                                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | caption             |     92 | 97%       | 39%     | 100       | 74        | 186       | 4.4       | clean                                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | caption             |     92 | 97%       | 46%     | 100       | 74        | 15.3      | 33        | clean                                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | caption             |     92 | 97%       | 28%     | 80        | 93        | 34.1      | 19        | formatting                            |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | avoid               |     92 | 97%       | 35%     | 100       | 66        | 27.2      | 19        | clean                                 |
| `qnguyen3/nanoLLaVA`                                    | caption-review      |     92 | 96%       | 33%     | 100       | 96        | 114       | 4.0       | clean                                 |
| `mlx-community/gemma-4-31b-bf16`                        | caption-review      |     69 | 94%       | 22%     | 100       | 96        | 7.1       | 64        | clean                                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | caption-review      |     92 | 96%       | 26%     | 100       | 88        | 60.7      | 9.7       | clean                                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | caption-review      |     92 | 96%       | 33%     | 100       | 77        | 344       | 1.9       | clean                                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | avoid               |     92 | 97%       | 25%     | 85        | 62        | 41.9      | 17        | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | caption-review      |     76 | 93%       | 22%     | 100       | 85        | 115       | 22        | clean                                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | avoid               |     92 | 96%       | 27%     | 85        | 62        | 130       | 5.5       | clean                                 |
| `mlx-community/InternVL3-8B-bf16`                       | caption-review      |     92 | 94%       | 39%     | 100       | 89        | 33.2      | 18        | clean                                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | caption-review      |     92 | 94%       | 29%     | 100       | 86        | 30.2      | 27        | clean                                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | avoid               |     92 | 97%       | 28%     | 50        | 62        | 78.8      | 16        | reasoning-leak                        |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | caption-review      |     92 | 92%       | 26%     | 100       | 88        | 68.8      | 71        | clean                                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | caption-review      |     92 | 92%       | 16%     | 100       | 88        | 314       | 2.5       | clean                                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | caption-review      |     92 | 92%       | 29%     | 100       | 85        | 65.2      | 10        | clean                                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | caption-review      |     92 | 91%       | 28%     | 100       | 96        | 17.7      | 30        | clean                                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | caption-review      |     92 | 91%       | 29%     | 100       | 90        | 331       | 2.0       | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | caption-review      |     92 | 91%       | 17%     | 100       | 86        | 97.8      | 30        | clean                                 |
| `mlx-community/Qwen3.5-27B-4bit`                        | caption-review      |     92 | 91%       | 16%     | 100       | 82        | 29.4      | 19        | clean                                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             | avoid               |     57 | 88%       | 30%     | 85        | 62        | 134       | 5.1       | clean                                 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | avoid               |     39 | 69%       | 44%     | 100       | 69        | 551       | 1.0       | clean                                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | avoid               |     92 | 88%       | 30%     | 85        | 62        | 18.9      | 10        | clean                                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | avoid               |     92 | 88%       | 28%     | 85        | 62        | 5         | 26        | clean                                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | avoid               |     92 | 88%       | 28%     | 85        | 62        | 31.0      | 11        | clean                                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | avoid               |     92 | 83%       | 28%     | 100       | 69        | 318       | 3.0       | clean                                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | avoid               |     92 | 88%       | 0%      | 15        | 29        | 77.0      | 4.6       | harness:long-context; context-budget  |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | avoid               |     92 | 97%       | 0%      | 0         | 74        | 33.1      | 19        | harness:encoding                      |
| `mlx-community/pixtral-12b-8bit`                        | avoid               |     92 | 97%       | 13%     | 0         | 62        | 37.0      | 15        | cutoff                                |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | avoid               |     92 | 97%       | 9%      | 0         | 62        | 4.3       | 39        | cutoff; reasoning-leak                |
| `microsoft/Phi-3.5-vision-instruct`                     | avoid               |     92 | 97%       | 0%      | 0         | 59        | 58.5      | 9.2       | harness:stop-token                    |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | avoid               |     92 | 97%       | 0%      | 0         | 54        | 44.3      | 14        | harness:stop-token; reasoning-leak    |
| `mlx-community/gemma-3n-E2B-4bit`                       | avoid               |     92 | 97%       | 0%      | 0         | 39        | 114       | 5.9       | repetitive; cutoff                    |
| `mlx-community/pixtral-12b-bf16`                        | avoid               |     92 | 96%       | 14%     | 0         | 62        | 19.3      | 27        | cutoff                                |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | avoid               |     92 | 89%       | 0%      | 0         | 54        | 64.7      | 60        | cutoff; degeneration                  |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | avoid               |     56 | 88%       | 0%      | 0         | 62        | 130       | 5.3       | harness:stop-token                    |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | avoid               |     92 | 63%       | 2%      | 0         | 62        | 64.2      | 20        | cutoff; reasoning-leak                |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | avoid               |     92 | 53%       | 10%     | 0         | -         | -         | -         | mlx; weight-mismatch                  |
| `prince-canuma/Florence-2-large-ft`                     | avoid               |     17 | 53%       | 0%      | -         | -         | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL         |
| `mlx-community/MolmoPoint-8B-fp16`                      | unstable            |     85 | 15%       | 8%      | 0         | -         | -         | -         | mlx-vlm; model-error                  |
| `mlx-community/granite-4.1-8b-mxfp8`                    | integration-blocked |      1 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                       |
| `mlx-community/deepseek-vl2-8bit`                       | integration-blocked |     11 | 0%        | 0%      | -         | -         | -         | -         | MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR |
| `microsoft/Florence-2-large-ft`                         | integration-blocked |     15 | 0%        | 0%      | -         | -         | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL         |
| `ggml-org/gemma-3-1b-it-GGUF`                           | integration-blocked |     11 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                       |
| `facebook/pe-av-large`                                  | integration-blocked |     21 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                       |
<!-- markdownlint-enable MD034 -->

Scores aggregate available current/history signals. Older history rows that
predate capability fields still count toward reliability, but not caption,
keyword, or metadata averages.
