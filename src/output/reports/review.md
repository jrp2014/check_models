# Automated Review Digest

_Generated on 2026-04-26 20:52:02 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/gemma-3-27b-it-qat-4bit`: 🏆 A (85/100) | Desc 82 | Keywords 92 | 30.8 tps
- `mlx-community/Qwen3.5-35B-A3B-bf16`: ✅ B (80/100) | Desc 81 | Keywords 90 | 64.0 tps
- `mlx-community/Molmo-7B-D-0924-8bit`: ✅ B (80/100) | Desc 84 | Keywords 0 | 52.1 tps
- `mlx-community/Molmo-7B-D-0924-bf16`: ✅ B (80/100) | Desc 84 | Keywords 0 | 30.4 tps
- `mlx-community/gemma-3-27b-it-qat-8bit`: ✅ B (80/100) | Desc 82 | Keywords 94 | 17.7 tps

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 40 | Keywords 0 | 31.9 tps | harness
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (5/100) | Desc 22 | Keywords 0 | 83.4 tps | harness, long context
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (5/100) | Desc 23 | Keywords 0 | 93.8 tps | harness
- `mlx-community/gemma-4-31b-bf16`: ❌ F (5/100) | Desc 23 | Keywords 0 | 9.3 tps | harness
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (16/100) | Desc 48 | Keywords 0 | 207.2 tps | harness, long context

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict     | Hint Handling           | Key Evidence                                                                                     |
|-----------------------------------------------------|-------------|-------------------------|--------------------------------------------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `clean`     | preserves trusted hints | nontext prompt burden=80%                                                                        |
| `qnguyen3/nanoLLaVA`                                | `clean`     | preserves trusted hints | nontext prompt burden=80%                                                                        |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                 | `clean`     | preserves trusted hints | nontext prompt burden=95%                                                                        |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/LFM2-VL-1.6B-8bit`                   | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/FastVLM-0.5B-bf16`                   | `clean`     | preserves trusted hints | nontext prompt burden=83%                                                                        |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`          | `clean`     | preserves trusted hints | nontext prompt burden=96%                                                                        |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/llava-v1.6-mistral-7b-8bit`          | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `clean`     | preserves trusted hints | nontext prompt burden=100% \| reasoning leak                                                     |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/SmolVLM-Instruct-bf16`               | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `clean`     | preserves trusted hints | nontext prompt burden=100% \| reasoning leak                                                     |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/gemma-3n-E4B-it-bf16`                | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`           | `token_cap` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                              |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/pixtral-12b-8bit`                    | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`     | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/pixtral-12b-bf16`                    | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/InternVL3-8B-bf16`                   | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/Molmo-7B-D-0924-8bit`                | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `clean`     | preserves trusted hints | nontext prompt burden=71%                                                                        |
| `mlx-community/X-Reasoner-7B-8bit`                  | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/Molmo-7B-D-0924-bf16`                | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `clean`     | preserves trusted hints | nontext prompt burden=73%                                                                        |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `token_cap` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100% \| degeneration=repeated_punctuation: ':**...' |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `token_cap` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                |

### `caveat`

| Model                                     | Verdict          | Hint Handling           | Key Evidence                                                                                                                                                                                                                                                        |
|-------------------------------------------|------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/paligemma2-3b-pt-896-4bit` | `context_budget` | preserves trusted hints | Output appears truncated to about 3 tokens. \| At long prompt length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%). \| output/prompt=0.07% \| nontext prompt burden=100%                                                                      |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | preserves trusted hints | Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues. \| At long prompt length (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%). \| output/prompt=0.08% \| nontext prompt burden=100% |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict           | Hint Handling           | Key Evidence                                                                                                                                                                                     |
|---------------------------------------------------------|-------------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure` | not evaluated           | processor error \| model config processor load processor                                                                                                                                         |
| `mlx-community/gemma-3n-E2B-4bit`                       | `harness`         | preserves trusted hints | Output appears truncated to about 4 tokens. \| nontext prompt burden=98%                                                                                                                         |
| `mlx-community/gemma-4-31b-bf16`                        | `harness`         | preserves trusted hints | Output appears truncated to about 5 tokens. \| nontext prompt burden=98%                                                                                                                         |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`         | preserves trusted hints | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 112 occurrences). \| nontext prompt burden=100%                                                                       |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`         | preserves trusted hints | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=99% |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `harness`         | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100% \| reasoning leak \| degeneration=incomplete_sentence: ends with 'of'                                                                          |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `harness`         | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          |
| `mlx-community/GLM-4.6V-nvfp4`                          | `harness`         | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `mlx`

| Model                                     | Verdict          | Evidence                                                                                                                                                                                                                                                            | Next Action                                                                                                         |
|-------------------------------------------|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `mlx-community/paligemma2-3b-pt-896-4bit` | `context_budget` | Output appears truncated to about 3 tokens. \| At long prompt length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%). \| output/prompt=0.07% \| nontext prompt burden=100%                                                                      | Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues. \| At long prompt length (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%). \| output/prompt=0.08% \| nontext prompt burden=100% | Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load. |

### `mlx-vlm`

| Model                                                   | Verdict   | Evidence                                                                                                                                                                                         | Next Action                                                                         |
|---------------------------------------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 112 occurrences). \| nontext prompt burden=100%                                                                       | Inspect decode cleanup; tokenizer markers are leaking into user-facing text.        |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=99% | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-nvfp4`                          | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model                                               | Verdict           | Evidence                                                                                                                | Next Action                                                    |
|-----------------------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `clean`           | nontext prompt burden=80%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `qnguyen3/nanoLLaVA`                                | `clean`           | nontext prompt burden=80%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                 | `clean`           | nontext prompt burden=95%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        | `clean`           | nontext prompt burden=99%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/LFM2-VL-1.6B-8bit`                   | `clean`           | nontext prompt burden=99%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/FastVLM-0.5B-bf16`                   | `clean`           | nontext prompt burden=83%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`          | `clean`           | nontext prompt burden=96%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`           | nontext prompt burden=99%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/llava-v1.6-mistral-7b-8bit`          | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `clean`           | nontext prompt burden=100% \| reasoning leak                                                                            | Treat as a model-quality limitation for this prompt and image. |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/SmolVLM-Instruct-bf16`               | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `clean`           | nontext prompt burden=100% \| reasoning leak                                                                            | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/gemma-3n-E4B-it-bf16`                | `clean`           | nontext prompt burden=99%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`           | `token_cap`       | hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                     | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/pixtral-12b-8bit`                    | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`     | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`           | nontext prompt burden=99%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/pixtral-12b-bf16`                    | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/InternVL3-8B-bf16`                   | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `clean`           | nontext prompt burden=99%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`     | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100% \| reasoning leak \| degeneration=incomplete_sentence: ends with 'of' | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`  | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Molmo-7B-D-0924-8bit`                | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `clean`           | nontext prompt burden=71%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/X-Reasoner-7B-8bit`                  | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `clean`           | nontext prompt burden=99%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Molmo-7B-D-0924-bf16`                | `clean`           | nontext prompt burden=100%                                                                                              | Treat as a model-quality limitation for this prompt and image. |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `clean`           | nontext prompt burden=73%                                                                                               | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `token_cap`       | hit token cap (500) \| nontext prompt burden=100% \| degeneration=repeated_punctuation: ':**...'                        | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `token_cap`       | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-27B-4bit`                    | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.6-27B-mxfp8`                   | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |

### `model-config`

| Model                              | Verdict           | Evidence                                                                 | Next Action                                                 |
|------------------------------------|-------------------|--------------------------------------------------------------------------|-------------------------------------------------------------|
| `mlx-community/MolmoPoint-8B-fp16` | `runtime_failure` | processor error \| model config processor load processor                 | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/gemma-3n-E2B-4bit`  | `harness`         | Output appears truncated to about 4 tokens. \| nontext prompt burden=98% | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/gemma-4-31b-bf16`   | `harness`         | Output appears truncated to about 5 tokens. \| nontext prompt burden=98% | Inspect model repo config, chat template, and EOS settings. |

## Model Verdicts

### `mlx-community/MolmoPoint-8B-fp16`

- **Verdict:** runtime_failure | user=avoid
- **Why:** processor error | model config processor load processor
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=model-config | package=model-config | stage=Processor Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** harness | user=avoid
- **Why:** Output appears truncated to about 4 tokens. | nontext prompt burden=98%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=264 | text_est=4 | nontext_est=260 | gen=4 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=80%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=20 | text_est=4 | nontext_est=16 | gen=124 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `qnguyen3/nanoLLaVA`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=80%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=20 | text_est=4 | nontext_est=16 | gen=51 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=95%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=85 | text_est=4 | nontext_est=81 | gen=117 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=766 | text_est=4 | nontext_est=762 | gen=41 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=275 | text_est=4 | nontext_est=271 | gen=292 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** Output appears truncated to about 3 tokens. | At long prompt length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%). | output/prompt=0.07% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=caveat | preserves trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=4101 | text_est=4 | nontext_est=4097 | gen=3 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=83%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=24 | text_est=4 | nontext_est=20 | gen=445 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-4-31b-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output appears truncated to about 5 tokens. | nontext prompt burden=98%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=266 | text_est=4 | nontext_est=262 | gen=5 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=96%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=95 | text_est=4 | nontext_est=91 | gen=266 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2326 | text_est=4 | nontext_est=2322 | gen=48 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-4-26b-a4b-it-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=278 | text_est=4 | nontext_est=274 | gen=267 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2629 | text_est=4 | nontext_est=2625 | gen=355 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2174 | text_est=4 | nontext_est=2170 | gen=74 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1011 | text_est=4 | nontext_est=1007 | gen=345 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1195 | text_est=4 | nontext_est=1191 | gen=420 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1195 | text_est=4 | nontext_est=1191 | gen=420 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1011 | text_est=4 | nontext_est=1007 | gen=296 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1029 | text_est=4 | nontext_est=1025 | gen=135 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=272 | text_est=4 | nontext_est=268 | gen=237 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** token_cap | user=recommended
- **Why:** hit token cap (500) | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1011 | text_est=4 | nontext_est=1007 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1804 | text_est=4 | nontext_est=1800 | gen=125 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2887 | text_est=4 | nontext_est=2883 | gen=167 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1029 | text_est=4 | nontext_est=1025 | gen=118 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-4-31b-it-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=278 | text_est=4 | nontext_est=274 | gen=158 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2630 | text_est=4 | nontext_est=2626 | gen=311 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 112 occurrences). | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | generic
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2098 | text_est=4 | nontext_est=2094 | gen=135 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2630 | text_est=4 | nontext_est=2626 | gen=344 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|end|&gt; appeared in generated text. | Special control token &lt;|endoftext|&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=766 | text_est=4 | nontext_est=762 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2887 | text_est=4 | nontext_est=2883 | gen=166 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1804 | text_est=4 | nontext_est=1800 | gen=344 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=273 | text_est=4 | nontext_est=269 | gen=320 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6091 | text_est=4 | nontext_est=6087 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=100% | reasoning leak | degeneration=incomplete_sentence: ends with 'of'
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2979 | text_est=4 | nontext_est=2975 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1293 | text_est=4 | nontext_est=1289 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6091 | text_est=4 | nontext_est=6087 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1029 | text_est=4 | nontext_est=1025 | gen=100 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1207 | text_est=4 | nontext_est=1203 | gen=205 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=71%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=14 | text_est=4 | nontext_est=10 | gen=421 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16299 | text_est=4 | nontext_est=16295 | gen=219 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=273 | text_est=4 | nontext_est=269 | gen=381 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1207 | text_est=4 | nontext_est=1203 | gen=205 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6091 | text_est=4 | nontext_est=6087 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=73%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=15 | text_est=4 | nontext_est=11 | gen=276 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** token_cap | user=recommended
- **Why:** hit token cap (500) | nontext prompt burden=100% | degeneration=repeated_punctuation: ':**...'
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues. | At long prompt length (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%). | output/prompt=0.08% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=caveat | preserves trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16299 | text_est=4 | nontext_est=16295 | gen=13 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** token_cap | user=recommended
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen3.6-27B-mxfp8`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.
