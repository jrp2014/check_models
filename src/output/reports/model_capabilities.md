# Model Capability Scorecard

_Generated on 2026-06-28 02:18:30 BST_

- Mode: triage
- Grounding: not grounded; caption hygiene only
- Coverage: current run plus 100 prior history run(s).

Structured metadata and keyword capability: not evaluated in triage mode.

<!-- markdownlint-disable MD034 -->

| Model                                                   | Use                 |   Runs | Success   |   Capability | Caption   | Keyword   | Metadata   | Gen TPS   | Peak GB   | Latest signal                         |
|---------------------------------------------------------|---------------------|--------|-----------|--------------|-----------|-----------|------------|-----------|-----------|---------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | caption             |     52 | 100%      |          100 | 95        | -         | 25         | 104       | 17        | clean                                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | caption             |     18 | 100%      |          100 | 93        | -         | 40         | 239       | 3.0       | clean                                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         | caption             |      9 | 100%      |          100 | 92        | -         | 9          | 23.8      | 28        | clean                                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | caption             |     49 | 100%      |          100 | 89        | -         | 24         | 18.9      | 30        | clean                                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               | needs-review        |     14 | 100%      |           79 | 62        | -         | 7          | 126       | 5.2       | clean                                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          | caption             |     13 | 92%       |           50 | 95        | -         | 8          | 23.3      | 29        | clean                                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        | avoid               |     94 | 98%       |           33 | 88        | -         | 32         | 124       | 5.5       | clean                                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | avoid               |     92 | 97%       |           25 | 96        | -         | 36         | 9.4       | 15        | clean                                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | avoid               |     92 | 97%       |           25 | 96        | -         | 30         | 92.7      | 7.7       | clean                                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | avoid               |     92 | 97%       |           25 | 92        | -         | 21         | 61.9      | 9.2       | clean                                 |
| `mlx-community/InternVL3-14B-8bit`                      | avoid               |     92 | 97%       |           25 | 91        | -         | 25         | 32.2      | 19        | clean                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | avoid               |     92 | 97%       |           25 | 89        | -         | 20         | 46.5      | 10        | clean                                 |
| `mlx-community/gemma-4-31b-it-4bit`                     | avoid               |     68 | 96%       |           25 | 96        | -         | 25         | 15.1      | 19        | clean                                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   | avoid               |     92 | 97%       |           25 | 88        | -         | 32         | 127       | 5.5       | clean                                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | avoid               |     92 | 97%       |           25 | 86        | -         | 39         | 48.3      | 20        | clean                                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | avoid               |     92 | 97%       |           24 | 83        | -         | 24         | 62.7      | 10        | clean                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | avoid               |     92 | 97%       |           24 | 82        | -         | 19         | 40.0      | 9.7       | clean                                 |
| `mlx-community/GLM-4.6V-nvfp4`                          | avoid               |     92 | 97%       |           24 | 82        | -         | 24         | 33.3      | 63        | clean                                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | avoid               |     74 | 96%       |           24 | 80        | -         | 26         | 96.7      | 7.0       | clean                                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | avoid               |     92 | 97%       |           24 | 77        | -         | 39         | 5.1       | 25        | clean                                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | avoid               |     92 | 97%       |           23 | 74        | -         | 22         | 157       | 4.4       | clean                                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | avoid               |     92 | 97%       |           23 | 74        | -         | 22         | 10.5      | 33        | clean                                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | avoid               |     92 | 97%       |           23 | 93        | -         | 23         | 35.0      | 19        | formatting                            |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | avoid               |     92 | 97%       |           22 | 66        | -         | 29         | 18.4      | 19        | clean                                 |
| `qnguyen3/nanoLLaVA`                                    | avoid               |     92 | 96%       |           20 | 96        | -         | 59         | 112       | 3.8       | clean                                 |
| `mlx-community/gemma-4-31b-bf16`                        | avoid               |     67 | 94%       |           20 | 96        | -         | 30         | 6.3       | 64        | clean                                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | avoid               |     92 | 97%       |           20 | 62        | -         | 39         | 29.2      | 17        | clean                                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | avoid               |     92 | 96%       |           20 | 88        | -         | 25         | 57.7      | 9.7       | clean                                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | avoid               |     92 | 96%       |           19 | 77        | -         | 56         | 368       | 1.9       | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | avoid               |     74 | 93%       |           16 | 85        | -         | 26         | 116       | 22        | clean                                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | avoid               |     92 | 96%       |           16 | 62        | -         | 22         | 128       | 5.5       | clean                                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | avoid               |     92 | 97%       |           15 | 62        | -         | 61         | 76.3      | 15        | reasoning-leak                        |
| `mlx-community/InternVL3-8B-bf16`                       | avoid               |     92 | 94%       |           14 | 89        | -         | 24         | 30.3      | 18        | clean                                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | avoid               |     92 | 94%       |           14 | 87        | -         | 38         | 29.9      | 27        | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | avoid               |     92 | 92%       |           12 | 88        | -         | 26         | 67.9      | 71        | clean                                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | avoid               |     92 | 92%       |           12 | 88        | -         | 7          | 300       | 2.5       | clean                                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | avoid               |     92 | 92%       |           12 | 85        | -         | 23         | 63.4      | 10        | clean                                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | avoid               |     92 | 91%       |           11 | 96        | -         | 24         | 14.7      | 30        | clean                                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | avoid               |     92 | 91%       |           11 | 90        | -         | 13         | 324       | 2.1       | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | avoid               |     92 | 91%       |           11 | 86        | -         | 24         | 98.1      | 30        | clean                                 |
| `mlx-community/Qwen3.5-27B-4bit`                        | avoid               |     92 | 91%       |           11 | 82        | -         | 24         | 22.4      | 19        | clean                                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             | avoid               |     57 | 88%       |           10 | 62        | -         | 7          | 133       | 4.7       | clean                                 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | avoid               |     37 | 68%       |            7 | 69        | -         | 24         | 551       | 1.0       | clean                                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | avoid               |     92 | 88%       |            7 | 62        | -         | 35         | 17.7      | 10        | clean                                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | avoid               |     92 | 88%       |            7 | 62        | -         | 20         | 4.3       | 26        | clean                                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | avoid               |     92 | 88%       |            7 | 62        | -         | 19         | 23.1      | 11        | clean                                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | avoid               |     92 | 83%       |            5 | 69        | -         | 23         | 293       | 3.0       | clean                                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | avoid               |     92 | 88%       |            1 | 29        | -         | 5          | 69.6      | 4.6       | harness:long-context; context-budget  |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | avoid               |     92 | 97%       |            0 | 74        | -         | 20         | 33.0      | 19        | harness:encoding                      |
| `mlx-community/pixtral-12b-8bit`                        | avoid               |     92 | 97%       |            0 | 62        | -         | 22         | 31.1      | 15        | cutoff                                |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | avoid               |     92 | 97%       |            0 | 62        | -         | 59         | 3.5       | 39        | cutoff; reasoning-leak                |
| `microsoft/Phi-3.5-vision-instruct`                     | avoid               |     92 | 97%       |            0 | 59        | -         | 20         | 57.5      | 9.2       | harness:stop-token                    |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | avoid               |     92 | 97%       |            0 | 54        | -         | 27         | 44.2      | 14        | harness:stop-token; reasoning-leak    |
| `mlx-community/gemma-3n-E2B-4bit`                       | avoid               |     92 | 97%       |            0 | 39        | -         | 0          | 98.8      | 6.0       | repetitive; cutoff                    |
| `mlx-community/pixtral-12b-bf16`                        | avoid               |     92 | 96%       |            0 | 62        | -         | 22         | 17.6      | 27        | cutoff                                |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | avoid               |     92 | 89%       |            0 | 54        | -         | 20         | 63.8      | 60        | cutoff; degeneration                  |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | avoid               |     56 | 88%       |            0 | 62        | -         | 4          | 127       | 5.3       | harness:stop-token                    |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | avoid               |     92 | 63%       |            0 | 62        | -         | 0          | 46.3      | 20        | cutoff; reasoning-leak                |
| `prince-canuma/Florence-2-large-ft`                     | avoid               |     19 | 58%       |            0 | -         | -         | -          | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL         |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | avoid               |     92 | 55%       |            0 | -         | -         | -          | -         | -         | mlx; weight-mismatch                  |
| `mlx-community/MolmoPoint-8B-fp16`                      | unstable            |     83 | 16%       |            0 | -         | -         | 50         | -         | -         | mlx-vlm; model-error                  |
| `mlx-community/granite-4.1-8b-mxfp8`                    | integration-blocked |      1 | 0%        |            0 | -         | -         | -          | -         | -         | runtime_failure                       |
| `mlx-community/deepseek-vl2-8bit`                       | integration-blocked |     11 | 0%        |            0 | -         | -         | -          | -         | -         | MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR |
| `microsoft/Florence-2-large-ft`                         | integration-blocked |     17 | 0%        |            0 | -         | -         | -          | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL         |
| `ggml-org/gemma-3-1b-it-GGUF`                           | integration-blocked |     11 | 0%        |            0 | -         | -         | -          | -         | -         | runtime_failure                       |
| `facebook/pe-av-large`                                  | integration-blocked |     21 | 0%        |            0 | -         | -         | -          | -         | -         | runtime_failure                       |
<!-- markdownlint-enable MD034 -->

Scores aggregate available current/history signals. Older history rows that
predate capability fields still count toward reliability, but not caption,
keyword, or metadata averages.
