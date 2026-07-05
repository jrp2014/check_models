# Model Capability Scorecard

Generated on: 2026-07-05 01:34:36 BST

- Mode: triage
- Grounding: not grounded; caption hygiene only
- Coverage: current run plus 100 prior history run(s).

Structured metadata and keyword capability: not evaluated in triage mode.
Use the Clean, Hygiene, Caption, speed, and memory columns as a caption-review shortlist; visual correctness still requires grounded metadata or manual review.

<!-- markdownlint-disable MD034 -->

| Model                                                   | Use                 | Current    |   Runs | Success   | Clean   | Hygiene   | Caption   | Gen TPS   | Peak GB   | Latest signal                             |
|---------------------------------------------------------|---------------------|------------|--------|-----------|---------|-----------|-----------|-----------|-----------|-------------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | caption             | passed     |     57 | 100%      | 84%     | 100       | 95        | 113       | 17        | clean                                     |
| `mlx-community/Ornith-1.0-35B-bf16`                     | caption             | passed     |      4 | 100%      | 100%    | 100       | 92        | 60.7      | 71        | clean                                     |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | caption             | passed     |     23 | 96%       | 87%     | 100       | 93        | 267       | 3.0       | clean                                     |
| `HuggingFaceTB/SmolVLM-Instruct`                        | caption             | passed     |     94 | 98%       | 19%     | 100       | 88        | 127       | 5.5       | clean                                     |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | avoid               | avoid      |     54 | 100%      | 48%     | 67        | 60        | 24.6      | 30        | harness:prompt-template                   |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | caption             | passed     |     92 | 97%       | 32%     | 100       | 96        | 19.0      | 15        | clean                                     |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | caption             | passed     |     92 | 97%       | 21%     | 100       | 96        | 92.8      | 7.7       | clean                                     |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | caption             | passed     |     92 | 97%       | 32%     | 100       | 92        | 60.6      | 9.2       | clean                                     |
| `mlx-community/InternVL3-14B-8bit`                      | caption             | passed     |     92 | 97%       | 34%     | 100       | 91        | 33.1      | 19        | clean                                     |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | caption             | passed     |     92 | 97%       | 50%     | 100       | 90        | 62.4      | 10        | clean                                     |
| `mlx-community/gemma-4-31b-it-4bit`                     | caption             | passed     |     73 | 96%       | 60%     | 100       | 96        | 26.0      | 19        | clean                                     |
| `mlx-community/SmolVLM-Instruct-bf16`                   | caption             | passed     |     92 | 97%       | 20%     | 100       | 88        | 128       | 5.5       | clean                                     |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | caption             | passed     |     92 | 97%       | 33%     | 100       | 86        | 51.1      | 20        | clean                                     |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | caption             | passed     |     92 | 97%       | 51%     | 100       | 83        | 64.2      | 9.7       | clean                                     |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | caption             | passed     |     92 | 97%       | 20%     | 100       | 83        | 63.7      | 10        | clean                                     |
| `mlx-community/GLM-4.6V-nvfp4`                          | caption             | passed     |     92 | 97%       | 20%     | 100       | 82        | 48.6      | 63        | clean                                     |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | caption             | passed     |     92 | 97%       | 32%     | 100       | 77        | 5         | 25        | clean                                     |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | caption             | passed     |     92 | 97%       | 49%     | 100       | 74        | 15.1      | 33        | clean                                     |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | caption             | passed     |     92 | 97%       | 32%     | 80        | 93        | 33.8      | 19        | formatting                                |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | caption             | passed     |     92 | 97%       | 42%     | 95        | 74        | 191       | 4.5       | clean                                     |
| `qnguyen3/nanoLLaVA`                                    | caption             | passed     |     92 | 96%       | 36%     | 100       | 96        | 114       | 4.0       | clean                                     |
| `mlx-community/gemma-4-31b-bf16`                        | caption             | passed     |     72 | 94%       | 25%     | 100       | 96        | 7.4       | 64        | clean                                     |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | needs-review        | passed     |     92 | 97%       | 38%     | 100       | 66        | 27.6      | 19        | clean                                     |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | caption             | passed     |     92 | 96%       | 29%     | 100       | 88        | 61.0      | 9.7       | clean                                     |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | caption             | passed     |     92 | 96%       | 36%     | 100       | 77        | 336       | 1.9       | clean                                     |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | avoid               | avoid      |     79 | 96%       | 37%     | 86        | 85        | 95.2      | 7.3       | text-sanity=gibberish(mixed_script_noise) |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | caption             | passed     |     79 | 94%       | 25%     | 100       | 87        | 116       | 22        | clean                                     |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               | avoid               | avoid      |     19 | 100%      | 58%     | 57        | 53        | 138       | 5.2       | harness:prompt-template                   |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | needs-review        | passed     |     92 | 97%       | 28%     | 85        | 62        | 45.1      | 17        | clean                                     |
| `mlx-community/InternVL3-8B-bf16`                       | caption             | passed     |     92 | 94%       | 42%     | 100       | 89        | 33.9      | 18        | clean                                     |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         | caption             | passed     |     14 | 79%       | 71%     | 100       | 92        | 24.2      | 28        | clean                                     |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | caption             | passed     |     92 | 94%       | 33%     | 100       | 86        | 30.2      | 27        | clean                                     |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | needs-review        | passed     |     92 | 96%       | 30%     | 85        | 62        | 130       | 5.5       | clean                                     |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | caption-review      | passed     |     92 | 92%       | 20%     | 92        | 89        | 300       | 2.5       | clean                                     |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | caption-review      | passed     |     92 | 91%       | 32%     | 100       | 96        | 16.7      | 30        | clean                                     |
| `mlx-community/FastVLM-0.5B-bf16`                       | caption-review      | passed     |     92 | 91%       | 33%     | 100       | 90        | 328       | 2.1       | clean                                     |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          | caption-review      | passed     |     18 | 78%       | 72%     | 100       | 94        | 22.1      | 29        | clean                                     |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | caption-review      | passed     |     92 | 91%       | 21%     | 100       | 87        | 98.5      | 30        | clean                                     |
| `mlx-community/Qwen3.5-27B-4bit`                        | caption-review      | passed     |     92 | 91%       | 20%     | 100       | 78        | 31.6      | 19        | clean                                     |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | avoid               | passed     |     92 | 97%       | 32%     | 50        | 62        | 76.6      | 16        | reasoning-leak                            |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | avoid               | avoid      |     92 | 92%       | 27%     | 67        | 68        | 70.2      | 71        | harness:prompt-template                   |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | avoid               | passed     |     42 | 71%       | 48%     | 100       | 69        | 548       | 1.0       | clean                                     |
| `mlx-community/X-Reasoner-7B-8bit`                      | avoid               | avoid      |     92 | 92%       | 30%     | 67        | 68        | 73.6      | 10        | harness:prompt-template                   |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | avoid               | passed     |     92 | 88%       | 34%     | 85        | 62        | 19.2      | 10        | clean                                     |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | avoid               | passed     |     92 | 88%       | 32%     | 85        | 62        | 5.2       | 26        | clean                                     |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | avoid               | passed     |     92 | 88%       | 32%     | 85        | 62        | 32.0      | 11        | clean                                     |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | avoid               | passed     |     92 | 83%       | 32%     | 100       | 69        | 315       | 3.0       | clean                                     |
| `Qwen/Qwen3-VL-2B-Instruct`                             | avoid               | avoid      |     57 | 88%       | 32%     | 57        | 53        | 119       | 5.0       | harness:prompt-template                   |
| `microsoft/Phi-3.5-vision-instruct`                     | caption-review      | passed     |     92 | 97%       | 2%      | 33        | 70        | 57.3      | 9.2       | clean                                     |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | caption-review      | passed     |     92 | 52%       | 12%     | 100       | 88        | 192       | 4.1       | clean                                     |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | avoid               | caveat     |     92 | 88%       | 0%      | 15        | 29        | 75.2      | 4.6       | harness:long-context; context-budget      |
| `mlx-community/MolmoPoint-8B-fp16`                      | unstable            | passed     |     88 | 17%       | 10%     | 85        | 62        | 5.9       | 23        | clean                                     |
| `mlx-community/pixtral-12b-8bit`                        | avoid               | avoid      |     92 | 97%       | 13%     | 0         | 62        | 38.2      | 15        | cutoff                                    |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | avoid               | avoid      |     92 | 97%       | 9%      | 0         | 62        | 4.5       | 39        | cutoff; reasoning-leak                    |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | avoid               | avoid      |     92 | 97%       | 0%      | 0         | 54        | 42.9      | 14        | harness:stop-token; reasoning-leak        |
| `mlx-community/gemma-3n-E2B-4bit`                       | avoid               | avoid      |     92 | 97%       | 0%      | 0         | 39        | 113       | 6.0       | repetitive; cutoff                        |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | current-run-blocked | failed     |     92 | 96%       | 0%      | 0         | 76        | 33.1      | 19        | huggingface-hub; model-error              |
| `mlx-community/pixtral-12b-bf16`                        | avoid               | avoid      |     92 | 96%       | 14%     | 0         | 62        | 19.7      | 27        | cutoff                                    |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | avoid               | avoid      |     92 | 89%       | 0%      | 0         | 54        | 64.6      | 60        | cutoff; degeneration                      |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | avoid               | avoid      |     56 | 88%       | 0%      | 0         | 68        | 131       | 5.2       | harness:stop-token                        |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | avoid               | avoid      |     92 | 63%       | 2%      | 0         | 62        | 68.3      | 20        | cutoff; reasoning-leak                    |
| `prince-canuma/Florence-2-large-ft`                     | unstable            | not-tested |     14 | 43%       | 0%      | -         | -         | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL             |
| `mlx-community/granite-4.1-8b-mxfp8`                    | integration-blocked | not-tested |      1 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                           |
| `mlx-community/deepseek-vl2-8bit`                       | integration-blocked | not-tested |     11 | 0%        | 0%      | -         | -         | -         | -         | MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR     |
| `microsoft/Florence-2-large-ft`                         | integration-blocked | not-tested |     14 | 0%        | 0%      | -         | -         | -         | -         | TRANSFORMERS_MODEL_LOAD_MODEL             |
| `ggml-org/gemma-3-1b-it-GGUF`                           | integration-blocked | not-tested |     11 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                           |
| `facebook/pe-av-large`                                  | integration-blocked | not-tested |     21 | 0%        | 0%      | -         | -         | -         | -         | runtime_failure                           |
<!-- markdownlint-enable MD034 -->

Scores aggregate available current/history signals. Older history rows that
predate capability fields still count toward reliability, but not caption,
keyword, or metadata averages.
