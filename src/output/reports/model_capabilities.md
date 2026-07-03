# Model Capability Scorecard

Generated on: 2026-07-04 00:01:55 BST

- Mode: triage
- Grounding: not grounded; caption hygiene only
- Coverage: current run plus 100 prior history run(s).

Structured metadata and keyword capability: not evaluated in triage mode.
Use the Clean, Hygiene, Caption, speed, and memory columns as a caption-review shortlist; visual correctness still requires grounded metadata or manual review.

<!-- markdownlint-disable MD034 -->

| Model                                                   | Use                 | Current    |   Runs | Success   | Clean   | Hygiene   | Caption   | Gen TPS   | Peak GB   | Latest signal                         |
|---------------------------------------------------------|---------------------|------------|--------|-----------|---------|-----------|-----------|-----------|-----------|---------------------------------------|
| `mlx-community/Ornith-1.0-35B-bf16`                     | caption             | passed     |      2 | 100%      | 100%    | 100       | 96        | 63.2      | 71        | clean                                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | caption             | passed     |     55 | 100%      | 84%     | 100       | 95        | 111       | 17        | clean                                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | caption             | passed     |     21 | 100%      | 90%     | 100       | 93        | 268       | 3.0       | clean                                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | caption             | passed     |     52 | 100%      | 50%     | 100       | 89        | 19.2      | 30        | clean                                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               | needs-review        | passed     |     17 | 100%      | 65%     | 85        | 62        | 131       | 5.2       | clean                                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        | caption             | passed     |     94 | 98%       | 17%     | 100       | 88        | 129       | 5.5       | clean                                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | caption             | passed     |     92 | 97%       | 29%     | 100       | 96        | 19.3      | 15        | clean                                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | caption             | passed     |     92 | 97%       | 18%     | 100       | 96        | 93.3      | 7.7       | clean                                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | caption             | passed     |     92 | 97%       | 29%     | 100       | 92        | 61.5      | 9.2       | clean                                 |
| `mlx-community/InternVL3-14B-8bit`                      | caption             | passed     |     92 | 97%       | 32%     | 100       | 91        | 32.9      | 19        | clean                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | caption             | passed     |     92 | 97%       | 48%     | 100       | 89        | 61.7      | 10        | clean                                 |
| `mlx-community/gemma-4-31b-it-4bit`                     | caption             | passed     |     71 | 96%       | 59%     | 100       | 96        | 25.0      | 19        | clean                                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   | caption             | passed     |     92 | 97%       | 17%     | 100       | 88        | 128       | 5.5       | clean                                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | caption             | passed     |     92 | 97%       | 30%     | 100       | 86        | 51.3      | 20        | clean                                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | caption             | passed     |     92 | 97%       | 17%     | 100       | 83        | 63.8      | 10        | clean                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | caption             | passed     |     92 | 97%       | 49%     | 100       | 82        | 62.7      | 9.7       | clean                                 |
| `mlx-community/GLM-4.6V-nvfp4`                          | caption             | passed     |     92 | 97%       | 17%     | 100       | 82        | 47.1      | 63        | clean                                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | caption             | passed     |     77 | 96%       | 36%     | 100       | 80        | 99.1      | 7.0       | clean                                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | caption             | passed     |     92 | 97%       | 29%     | 100       | 77        | 5.1       | 25        | clean                                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | caption             | passed     |     92 | 97%       | 40%     | 100       | 74        | 191       | 4.4       | clean                                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | caption             | passed     |     92 | 97%       | 47%     | 100       | 74        | 14.6      | 33        | clean                                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | caption             | passed     |     92 | 97%       | 29%     | 80        | 93        | 34.0      | 19        | formatting                            |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | needs-review        | passed     |     92 | 97%       | 36%     | 100       | 66        | 27.0      | 19        | clean                                 |
| `qnguyen3/nanoLLaVA`                                    | caption             | passed     |     92 | 96%       | 34%     | 100       | 96        | 113       | 4.0       | clean                                 |
| `mlx-community/gemma-4-31b-bf16`                        | caption             | passed     |     70 | 94%       | 23%     | 100       | 96        | 7.2       | 64        | clean                                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         | current-run-blocked | failed     |     12 | 83%       | 75%     | 67        | 93        | 26.1      | 28        | model-config; processor-error         |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | caption             | passed     |     92 | 96%       | 27%     | 100       | 88        | 60.9      | 9.7       | clean                                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | caption             | passed     |     92 | 96%       | 34%     | 100       | 77        | 332       | 1.9       | clean                                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | needs-review        | passed     |     92 | 97%       | 26%     | 85        | 62        | 43.4      | 17        | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | caption-review      | passed     |     77 | 94%       | 23%     | 100       | 85        | 116       | 22        | clean                                 |
| `mlx-community/InternVL3-8B-bf16`                       | caption-review      | passed     |     92 | 94%       | 40%     | 100       | 89        | 33.6      | 18        | clean                                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          | current-run-blocked | failed     |     16 | 81%       | 75%     | 67        | 94        | 20.6      | 29        | model-config; processor-error         |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | avoid               | passed     |     92 | 96%       | 28%     | 85        | 62        | 130       | 5.5       | clean                                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | caption-review      | passed     |     92 | 94%       | 30%     | 100       | 86        | 30.3      | 27        | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | caption-review      | passed     |     92 | 92%       | 27%     | 100       | 88        | 68.3      | 71        | clean                                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | caption-review      | passed     |     92 | 92%       | 17%     | 100       | 88        | 317       | 2.5       | clean                                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | caption-review      | passed     |     92 | 92%       | 30%     | 100       | 85        | 65.2      | 10        | clean                                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | avoid               | passed     |     92 | 97%       | 29%     | 50        | 62        | 79.0      | 16        | reasoning-leak                        |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | caption-review      | passed     |     92 | 91%       | 29%     | 100       | 96        | 18.1      | 30        | clean                                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | caption-review      | passed     |     92 | 91%       | 30%     | 100       | 90        | 331       | 2.1       | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | caption-review      | passed     |     92 | 91%       | 18%     | 100       | 86        | 98.7      | 30        | clean                                 |
| `mlx-community/Qwen3.5-27B-4bit`                        | caption-review      | passed     |     92 | 91%       | 17%     | 100       | 82        | 30.5      | 19        | clean                                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             | avoid               | passed     |     57 | 88%       | 32%     | 85        | 62        | 135       | 5.1       | clean                                 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | avoid               | passed     |     40 | 70%       | 45%     | 100       | 69        | 554       | 1.0       | clean                                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | avoid               | passed     |     92 | 88%       | 32%     | 85        | 62        | 19.1      | 10        | clean                                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | avoid               | passed     |     92 | 88%       | 29%     | 85        | 62        | 5.1       | 26        | clean                                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | avoid               | passed     |     92 | 88%       | 29%     | 85        | 62        | 31.3      | 11        | clean                                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | avoid               | passed     |     92 | 83%       | 29%     | 100       | 69        | 320       | 3.0       | clean                                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | avoid               | caveat     |     92 | 88%       | 0%      | 15        | 29        | 76.1      | 4.6       | harness:long-context; context-budget  |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | avoid               | avoid      |     92 | 97%       | 0%      | 0         | 74        | 33.1      | 19        | harness:encoding                      |
| `mlx-community/pixtral-12b-8bit`                        | avoid               | avoid      |     92 | 97%       | 13%     | 0         | 62        | 37.8      | 15        | cutoff                                |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | avoid               | avoid      |     92 | 97%       | 9%      | 0         | 62        | 4.4       | 39        | cutoff; reasoning-leak                |
| `microsoft/Phi-3.5-vision-instruct`                     | avoid               | avoid      |     92 | 97%       | 0%      | 0         | 59        | 58.3      | 9.2       | harness:stop-token                    |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | avoid               | avoid      |     92 | 97%       | 0%      | 0         | 54        | 44.4      | 14        | harness:stop-token; reasoning-leak    |
| `mlx-community/gemma-3n-E2B-4bit`                       | avoid               | avoid      |     92 | 97%       | 0%      | 0         | 39        | 108       | 6.0       | repetitive; cutoff                    |
| `mlx-community/pixtral-12b-bf16`                        | avoid               | avoid      |     92 | 96%       | 14%     | 0         | 62        | 19.5      | 27        | cutoff                                |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | avoid               | avoid      |     92 | 89%       | 0%      | 0         | 54        | 64.9      | 60        | cutoff; degeneration                  |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | avoid               | avoid      |     56 | 88%       | 0%      | 0         | 62        | 131       | 5.3       | harness:stop-token                    |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | avoid               | avoid      |     92 | 63%       | 2%      | 0         | 62        | 66.8      | 20        | cutoff; reasoning-leak                |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | current-run-blocked | failed     |     92 | 52%       | 10%     | 0         | -         | -         | -         | mlx; weight-mismatch                  |
| `prince-canuma/Florence-2-large-ft`                     | avoid               | not-tested |     16 | 50%       | 0%      | -         | -         | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL         |
| `mlx-community/MolmoPoint-8B-fp16`                      | current-run-blocked | failed     |     86 | 15%       | 8%      | 0         | -         | -         | -         | mlx-vlm; model-error                  |
| `mlx-community/granite-4.1-8b-mxfp8`                    | integration-blocked | not-tested |      1 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                       |
| `mlx-community/deepseek-vl2-8bit`                       | integration-blocked | not-tested |     11 | 0%        | 0%      | -         | -         | -         | -         | MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR |
| `microsoft/Florence-2-large-ft`                         | integration-blocked | not-tested |     15 | 0%        | 0%      | -         | -         | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL         |
| `ggml-org/gemma-3-1b-it-GGUF`                           | integration-blocked | not-tested |     11 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                       |
| `facebook/pe-av-large`                                  | integration-blocked | not-tested |     21 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                       |
<!-- markdownlint-enable MD034 -->

Scores aggregate available current/history signals. Older history rows that
predate capability fields still count toward reliability, but not caption,
keyword, or metadata averages.
