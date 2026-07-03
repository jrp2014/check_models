# Model Capability Scorecard

Generated on: 2026-06-28 22:11:08 BST

- Mode: triage
- Grounding: not grounded; caption hygiene only
- Coverage: current run plus 100 prior history run(s).

Structured metadata and keyword capability: not evaluated in triage mode.

<!-- markdownlint-disable MD034 -->

| Model                                                   | Use                 |   Runs | Success   |   Capability | Caption   | Keyword   | Metadata   | Gen TPS   | Peak GB   | Latest signal                         |
|---------------------------------------------------------|---------------------|--------|-----------|--------------|-----------|-----------|------------|-----------|-----------|---------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | caption             |     53 | 100%      |          100 | 95        | -         | 25         | 109       | 17        | clean                                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | caption             |     19 | 100%      |          100 | 93        | -         | 40         | 258       | 3.0       | clean                                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         | caption             |     10 | 100%      |          100 | 93        | -         | 9          | 26.1      | 28        | clean                                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | caption             |     50 | 100%      |          100 | 89        | -         | 24         | 19.1      | 30        | clean                                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               | needs-review        |     15 | 100%      |           79 | 62        | -         | 7          | 128       | 5.2       | clean                                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          | caption             |     14 | 93%       |           67 | 94        | -         | 8          | 20.6      | 29        | clean                                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        | caption             |     94 | 98%       |           50 | 88        | -         | 32         | 128       | 5.5       | clean                                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | avoid               |     92 | 97%       |           40 | 96        | -         | 36         | 15.9      | 15        | clean                                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | avoid               |     92 | 97%       |           40 | 96        | -         | 30         | 92.9      | 7.7       | clean                                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | avoid               |     92 | 97%       |           40 | 92        | -         | 21         | 61.0      | 9.2       | clean                                 |
| `mlx-community/InternVL3-14B-8bit`                      | avoid               |     92 | 97%       |           40 | 91        | -         | 25         | 32.7      | 19        | clean                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | avoid               |     92 | 97%       |           40 | 89        | -         | 20         | 56.0      | 10        | clean                                 |
| `mlx-community/gemma-4-31b-it-4bit`                     | avoid               |     69 | 96%       |           40 | 96        | -         | 25         | 21.8      | 19        | clean                                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   | avoid               |     92 | 97%       |           40 | 88        | -         | 32         | 128       | 5.5       | clean                                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | avoid               |     92 | 97%       |           39 | 86        | -         | 39         | 49.9      | 20        | clean                                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | avoid               |     92 | 97%       |           39 | 83        | -         | 24         | 63.2      | 10        | clean                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | avoid               |     92 | 97%       |           39 | 82        | -         | 19         | 54.7      | 9.7       | clean                                 |
| `mlx-community/GLM-4.6V-nvfp4`                          | avoid               |     92 | 97%       |           39 | 82        | -         | 24         | 42.2      | 63        | clean                                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | avoid               |     75 | 96%       |           38 | 80        | -         | 26         | 98.1      | 7.0       | clean                                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | avoid               |     92 | 97%       |           38 | 77        | -         | 39         | 5.1       | 25        | clean                                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | avoid               |     92 | 97%       |           37 | 74        | -         | 22         | 178       | 4.4       | clean                                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | avoid               |     92 | 97%       |           37 | 74        | -         | 22         | 14.0      | 33        | clean                                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | avoid               |     92 | 97%       |           36 | 93        | -         | 23         | 34.6      | 19        | formatting                            |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | avoid               |     92 | 97%       |           36 | 66        | -         | 29         | 24.9      | 19        | clean                                 |
| `qnguyen3/nanoLLaVA`                                    | avoid               |     92 | 96%       |           33 | 96        | -         | 59         | 113       | 3.9       | clean                                 |
| `mlx-community/gemma-4-31b-bf16`                        | avoid               |     68 | 94%       |           33 | 96        | -         | 30         | 6.9       | 64        | clean                                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | avoid               |     92 | 96%       |           33 | 88        | -         | 25         | 59.2      | 9.7       | clean                                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | avoid               |     92 | 97%       |           32 | 62        | -         | 39         | 38.5      | 17        | clean                                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | avoid               |     92 | 96%       |           32 | 77        | -         | 56         | 350       | 1.9       | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | avoid               |     75 | 93%       |           28 | 85        | -         | 26         | 113       | 22        | clean                                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | avoid               |     92 | 96%       |           26 | 62        | -         | 22         | 128       | 5.5       | clean                                 |
| `mlx-community/InternVL3-8B-bf16`                       | avoid               |     92 | 94%       |           25 | 89        | -         | 24         | 32.6      | 18        | clean                                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | avoid               |     92 | 94%       |           25 | 87        | -         | 38         | 29.9      | 27        | clean                                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | avoid               |     92 | 97%       |           24 | 62        | -         | 61         | 77.7      | 16        | reasoning-leak                        |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | avoid               |     92 | 92%       |           22 | 88        | -         | 26         | 67.9      | 71        | clean                                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | avoid               |     92 | 92%       |           22 | 88        | -         | 7          | 306       | 2.5       | clean                                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | avoid               |     92 | 92%       |           22 | 85        | -         | 23         | 64.6      | 10        | clean                                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | avoid               |     92 | 91%       |           20 | 96        | -         | 24         | 16.7      | 30        | clean                                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | avoid               |     92 | 91%       |           20 | 90        | -         | 13         | 327       | 2.1       | clean                                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | avoid               |     92 | 91%       |           20 | 86        | -         | 24         | 95.9      | 30        | clean                                 |
| `mlx-community/Qwen3.5-27B-4bit`                        | avoid               |     92 | 91%       |           19 | 82        | -         | 24         | 27.3      | 19        | clean                                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             | avoid               |     57 | 88%       |           18 | 62        | -         | 7          | 132       | 5.0       | clean                                 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | avoid               |     38 | 68%       |           13 | 69        | -         | 24         | 549       | 1.0       | clean                                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | avoid               |     92 | 88%       |           12 | 62        | -         | 35         | 18.5      | 10        | clean                                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | avoid               |     92 | 88%       |           12 | 62        | -         | 20         | 4.8       | 26        | clean                                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | avoid               |     92 | 88%       |           12 | 62        | -         | 19         | 28.7      | 11        | clean                                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | avoid               |     92 | 83%       |           10 | 69        | -         | 23         | 308       | 3.0       | clean                                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | avoid               |     92 | 88%       |            2 | 29        | -         | 5          | 73.3      | 4.6       | harness:long-context; context-budget  |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | avoid               |     92 | 97%       |            0 | 74        | -         | 20         | 33.0      | 19        | harness:encoding                      |
| `mlx-community/pixtral-12b-8bit`                        | avoid               |     92 | 97%       |            0 | 62        | -         | 22         | 35.5      | 15        | cutoff                                |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | avoid               |     92 | 97%       |            0 | 62        | -         | 59         | 4.1       | 39        | cutoff; reasoning-leak                |
| `microsoft/Phi-3.5-vision-instruct`                     | avoid               |     92 | 97%       |            0 | 59        | -         | 20         | 57.8      | 9.2       | harness:stop-token                    |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | avoid               |     92 | 97%       |            0 | 54        | -         | 27         | 44.2      | 14        | harness:stop-token; reasoning-leak    |
| `mlx-community/gemma-3n-E2B-4bit`                       | avoid               |     92 | 97%       |            0 | 39        | -         | 0          | 110       | 6.0       | repetitive; cutoff                    |
| `mlx-community/pixtral-12b-bf16`                        | avoid               |     92 | 96%       |            0 | 62        | -         | 22         | 19.0      | 27        | cutoff                                |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | avoid               |     92 | 89%       |            0 | 54        | -         | 20         | 64.4      | 60        | cutoff; degeneration                  |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | avoid               |     56 | 88%       |            0 | 62        | -         | 4          | 127       | 5.3       | harness:stop-token                    |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | avoid               |     92 | 63%       |            0 | 62        | -         | 0          | 59.3      | 20        | cutoff; reasoning-leak                |
| `prince-canuma/Florence-2-large-ft`                     | avoid               |     18 | 56%       |            0 | -         | -         | -          | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL         |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | avoid               |     92 | 54%       |            0 | -         | -         | -          | -         | -         | mlx; weight-mismatch                  |
| `mlx-community/MolmoPoint-8B-fp16`                      | unstable            |     84 | 16%       |            0 | -         | -         | 50         | -         | -         | mlx-vlm; model-error                  |
| `mlx-community/granite-4.1-8b-mxfp8`                    | integration-blocked |      1 | 0%        |            0 | -         | -         | -          | -         | -         | runtime_failure                       |
| `mlx-community/deepseek-vl2-8bit`                       | integration-blocked |     11 | 0%        |            0 | -         | -         | -          | -         | -         | MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR |
| `microsoft/Florence-2-large-ft`                         | integration-blocked |     16 | 0%        |            0 | -         | -         | -          | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL         |
| `ggml-org/gemma-3-1b-it-GGUF`                           | integration-blocked |     11 | 0%        |            0 | -         | -         | -          | -         | -         | runtime_failure                       |
| `facebook/pe-av-large`                                  | integration-blocked |     21 | 0%        |            0 | -         | -         | -          | -         | -         | runtime_failure                       |
<!-- markdownlint-enable MD034 -->

Scores aggregate available current/history signals. Older history rows that
predate capability fields still count toward reliability, but not caption,
keyword, or metadata averages.
