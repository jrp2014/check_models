# Automated Review Digest

_Generated on 2026-04-19 21:12:47 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (85/100) | Desc 93 | Keywords 89 | Δ+10 | 66.6 tps

### Watchlist

- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-75 | 296.6 tps | context ignored, harness
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) | Desc 44 | Keywords 0 | Δ-70 | 213.6 tps | context ignored, harness, metadata borrowing, missing sections
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | Δ-69 | 31.7 tps | harness, metadata borrowing, missing sections
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (6/100) | Desc 23 | Keywords 42 | Δ-69 | 22.8 tps | context ignored, harness
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (10/100) | Desc 47 | Keywords 42 | Δ-65 | 30.8 tps | context ignored, harness, missing sections

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict   | Hint Handling           | Key Evidence                                                                              |
|-----------------------------------------------------|-----------|-------------------------|-------------------------------------------------------------------------------------------|
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`   | preserves trusted hints | nontext prompt burden=85% \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger |

### `caveat`

| Model                                               | Verdict          | Hint Handling                                                                      | Key Evidence                                                                                                                                                                                           |
|-----------------------------------------------------|------------------|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `context_budget` | preserves trusted hints \| nonvisual metadata reused                               | output/prompt=4.60% \| nontext prompt burden=85% \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, People \| nonvisual metadata reused                                                             |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`          | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday | nontext prompt burden=80% \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `context_budget` | preserves trusted hints \| nonvisual metadata reused                               | output/prompt=3.90% \| nontext prompt burden=85% \| missing terms: Bird, Holiday, People, Riverbank, Stone wall \| nonvisual metadata reused                                                           |
| `mlx-community/pixtral-12b-8bit`                    | `context_budget` | preserves trusted hints \| nonvisual metadata reused                               | output/prompt=3.65% \| nontext prompt burden=86% \| missing terms: Bird, Blue sky \| keywords=19                                                                                                       |
| `mlx-community/X-Reasoner-7B-8bit`                  | `context_budget` | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday | At long prompt length (16780 tokens), output may stop following prompt/image context. \| output/prompt=0.44% \| nontext prompt burden=97% \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                                   | Key Evidence                                                                                                                                                                                                                                                             |
|---------------------------------------------------------|---------------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ggml-org/gemma-3-1b-it-GGUF`                           | `runtime_failure`   | not evaluated                                                                                                   | model error \| model config model load model                                                                                                                                                                                                                             |
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure`   | not evaluated                                                                                                   | processor error \| model config processor load processor                                                                                                                                                                                                                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | degrades trusted hints \| nonvisual metadata reused                                                             | missing sections: keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| nonvisual metadata reused                                                                                                                                                   |
| `mlx-community/FastVLM-0.5B-bf16`                       | `harness`           | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | Output appears truncated to about 8 tokens. \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                                                                                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                            | missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, People \| nonvisual metadata reused                                                                                                                                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `harness`           | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | Output appears truncated to about 6 tokens. \| nontext prompt burden=70% \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                                                    |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `harness`           | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday               |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | hit token cap (500) \| missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| keyword duplication=90% \| nonvisual metadata reused                                                                                                                               |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `harness`           | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                            | nontext prompt burden=66% \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger \| keywords=19 \| nonvisual metadata reused                                                                                                                                    |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                            | missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| nonvisual metadata reused                                                                                                                                                                                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                            | nontext prompt burden=83% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| nonvisual metadata reused                                                                                                  |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                                                       |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                          |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                          |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday \| nonvisual metadata reused                                                                                                          |
| `qnguyen3/nanoLLaVA`                                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                                                       |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                          |
| `mlx-community/gemma-4-31b-it-4bit`                     | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                            | missing terms: Berkshire, Bird, Holiday, Passenger, People \| nonvisual metadata reused                                                                                                                                                                                  |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                            | missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, People \| nonvisual metadata reused                                                                                                                                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                          |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                            | missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger \| nonvisual metadata reused                                                                                                                                                                                |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                            | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 61 occurrences). \| nontext prompt burden=82% \| missing sections: description, keywords \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger                                      |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                          |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                            | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=66%                                                                         |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Holiday, People, Quay                                                                                                              |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, keywords \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, People                                                                                                                       |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                            | Special control token &lt;/think&gt; appeared in generated text. \| nontext prompt burden=75% \| missing terms: Berkshire, Bird, Holiday, Passenger, People \| keywords=30                                                                                               |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                          |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, Sightseeing                                                                                                     |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | hit token cap (500) \| nontext prompt burden=84% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                          |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                            | nontext prompt burden=73% \| missing terms: Bird, Blue sky, filled, cruises, along \| keywords=19 \| nonvisual metadata reused                                                                                                                                           |
| `Qwen/Qwen3-VL-2B-Instruct`                             | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | At long prompt length (16769 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| keyword duplication=86%                                                                                                                           |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | `cutoff_degraded`   | degrades trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                          |
| `mlx-community/gemma-4-31b-bf16`                        | `model_shortcoming` | degrades trusted hints \| nonvisual metadata reused                                                             | missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| nonvisual metadata reused                                                                                                                               |
| `mlx-community/pixtral-12b-bf16`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                          |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| nonvisual metadata reused                                                                                                        |
| `mlx-community/InternVL3-8B-bf16`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | hit token cap (500) \| nontext prompt burden=80% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                          |
| `mlx-community/GLM-4.6V-nvfp4`                          | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: Berkshire, Bird, Holiday, People, Sightseeing                                                                                                                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `model_shortcoming` | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday \| repetitive token=phrase: "\*/ \*/ \*/ \*/..."                                                                                |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | hit token cap (500) \| nontext prompt burden=97% \| missing sections: description, keywords \| nonvisual metadata reused                                                                                                                                                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness`           | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday \| nonvisual metadata reused | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=97% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                          |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: filled, passengers, cruises, along, background                                                                                                      |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                              | At long prompt length (16794 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                                                              |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Passenger                                                                                                       |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Holiday, Lifebuoy, Sightseeing, Tree, filled                                                                                                        |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                            | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Sightseeing \| nonvisual metadata reused                                                                                                     |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `mlx`

| Model                                | Verdict           | Evidence                                                                                                                                                                                               | Next Action                                                                                                        |
|--------------------------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `mlx-community/X-Reasoner-7B-8bit`   | `context_budget`  | At long prompt length (16780 tokens), output may stop following prompt/image context. \| output/prompt=0.44% \| nontext prompt burden=97% \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday | Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load. |
| `Qwen/Qwen3-VL-2B-Instruct`          | `cutoff_degraded` | At long prompt length (16769 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| keyword duplication=86%                                                         | Inspect long-context cache behavior under heavy image-token burden.                                                |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff_degraded` | At long prompt length (16794 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords            | Inspect long-context cache behavior under heavy image-token burden.                                                |

### `mlx-vlm`

| Model                                                   | Verdict   | Evidence                                                                                                                                                                                                                                                                 | Next Action                                                                         |
|---------------------------------------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 61 occurrences). \| nontext prompt burden=82% \| missing sections: description, keywords \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger                                      | Inspect decode cleanup; tokenizer markers are leaking into user-facing text.        |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=66%                                                                         | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| nontext prompt burden=75% \| missing terms: Berkshire, Bird, Holiday, Passenger, People \| keywords=30                                                                                               | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness` | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=97% \| missing sections: title, description, keywords | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model                                               | Verdict             | Evidence                                                                                                                                                                              | Next Action                                                                                                        |
|-----------------------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `model_shortcoming` | missing sections: keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| nonvisual metadata reused                                                                | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/LFM2-VL-1.6B-8bit`                   | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, People \| nonvisual metadata reused                                              | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `context_budget`    | output/prompt=4.60% \| nontext prompt burden=85% \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, People \| nonvisual metadata reused                                            | Treat this as a prompt-budget issue first; nontext prompt burden is 85% and the output stays weak under that load. |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                 | `cutoff_degraded`   | hit token cap (500) \| missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| keyword duplication=90% \| nonvisual metadata reused                                            | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        | `model_shortcoming` | nontext prompt burden=66% \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger \| keywords=19 \| nonvisual metadata reused                                                 | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`             | nontext prompt burden=85% \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger                                                                                             | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `model_shortcoming` | missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| nonvisual metadata reused                                                                                              | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/llava-v1.6-mistral-7b-8bit`          | `model_shortcoming` | nontext prompt burden=83% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| nonvisual metadata reused               | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`          | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`             | nontext prompt burden=80% \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `context_budget`    | output/prompt=3.90% \| nontext prompt burden=85% \| missing terms: Bird, Holiday, People, Riverbank, Stone wall \| nonvisual metadata reused                                          | Treat this as a prompt-budget issue first; nontext prompt burden is 85% and the output stays weak under that load. |
| `mlx-community/SmolVLM-Instruct-bf16`               | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-3n-E2B-4bit`                   | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday \| nonvisual metadata reused                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `qnguyen3/nanoLLaVA`                                | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-4-31b-it-4bit`                 | `model_shortcoming` | missing terms: Berkshire, Bird, Holiday, Passenger, People \| nonvisual metadata reused                                                                                               | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/pixtral-12b-8bit`                    | `context_budget`    | output/prompt=3.65% \| nontext prompt burden=86% \| missing terms: Bird, Blue sky \| keywords=19                                                                                      | Treat this as a prompt-budget issue first; nontext prompt burden is 86% and the output stays weak under that load. |
| `mlx-community/gemma-3n-E4B-it-bf16`                | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, People \| nonvisual metadata reused                                              | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`           | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `model_shortcoming` | missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger \| nonvisual metadata reused                                                                                             | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Holiday, People, Quay                           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`     | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, keywords \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, People                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/paligemma2-3b-pt-896-4bit`           | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-Flash-6bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Holiday, Lifebuoy, Sightseeing                  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=84% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Molmo-7B-D-0924-8bit`                | `model_shortcoming` | nontext prompt burden=73% \| missing terms: Bird, Blue sky, filled, cruises, along \| keywords=19 \| nonvisual metadata reused                                                        | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`           | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-4-31b-bf16`                    | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| nonvisual metadata reused                                            | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/pixtral-12b-bf16`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy \| nonvisual metadata reused                     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/InternVL3-8B-bf16`                   | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=80% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-nvfp4`                      | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: Berkshire, Bird, Holiday, People, Sightseeing                              | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Molmo-7B-D-0924-bf16`                | `model_shortcoming` | nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday \| repetitive token=phrase: "\*/ \*/..."     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: description, keywords \| nonvisual metadata reused                                                              | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: filled, passengers, cruises, along, background                   | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-4bit`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Passenger                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Holiday, Lifebuoy, Sightseeing, Tree, filled                     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Holiday, Sightseeing \| nonvisual metadata reused                  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |

### `model-config`

| Model                                            | Verdict           | Evidence                                                                                                                                                                                                                                                   | Next Action                                                                                          |
|--------------------------------------------------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `ggml-org/gemma-3-1b-it-GGUF`                    | `runtime_failure` | model error \| model config model load model                                                                                                                                                                                                               | Inspect model repo config, chat template, and EOS settings.                                          |
| `mlx-community/MolmoPoint-8B-fp16`               | `runtime_failure` | processor error \| model config processor load processor                                                                                                                                                                                                   | Inspect model repo config, chat template, and EOS settings.                                          |
| `mlx-community/FastVLM-0.5B-bf16`                | `harness`         | Output appears truncated to about 8 tokens. \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                                                                   | Inspect model repo config, chat template, and EOS settings.                                          |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`  | `harness`         | Output appears truncated to about 6 tokens. \| nontext prompt burden=70% \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                                                                                      | Inspect model repo config, chat template, and EOS settings.                                          |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `harness`         | Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday | Check chat-template and EOS defaults first; the output shape is not matching the requested contract. |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness`         | Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing terms: Berkshire, Bird, Blue sky, Castle, Holiday                                                   | Inspect model repo config, chat template, and EOS settings.                                          |

## Model Verdicts

### `ggml-org/gemma-3-1b-it-GGUF`

- **Verdict:** runtime_failure | user=avoid
- **Why:** model error | model config model load model
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=model-config | package=model-config | stage=Model Error | code=MODEL_CONFIG_MODEL_LOAD_MODEL
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/MolmoPoint-8B-fp16`

- **Verdict:** runtime_failure | user=avoid
- **Why:** processor error | model config processor load processor
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=model-config | package=model-config | stage=Processor Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: keywords | missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy | nonvisual metadata reused
- **Trusted hints:** degrades trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords
- **Utility:** user=avoid | degrades trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=552 | text_est=466 | nontext_est=86 | gen=51 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output appears truncated to about 8 tokens. | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=556 | text_est=466 | nontext_est=90 | gen=8 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Holiday, People | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=812 | text_est=466 | nontext_est=346 | gen=166 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output appears truncated to about 6 tokens. | nontext prompt burden=70% | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1574 | text_est=466 | nontext_est=1108 | gen=6 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=70% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1574 | text_est=466 | nontext_est=1108 | gen=12 | max=500 | stop=completed
- **Next action:** Check chat-template and EOS defaults first; the output shape is not matching the requested contract.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=4.60% | nontext prompt burden=85% | missing terms: Berkshire, Bird, Holiday, Lifebuoy, People | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=caveat | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3151 | text_est=466 | nontext_est=2685 | gen=145 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 85% and the output stays weak under that load.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy | keyword duplication=90% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=197 | keyword duplication=0.90
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=622 | text_est=466 | nontext_est=156 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=70% | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1574 | text_est=466 | nontext_est=1108 | gen=9 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=66% | missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger | keywords=19 | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=19
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1382 | text_est=466 | nontext_est=916 | gen=180 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=85% | missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3152 | text_est=466 | nontext_est=2686 | gen=109 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=820 | text_est=466 | nontext_est=354 | gen=81 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=83% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2764 | text_est=466 | nontext_est=2298 | gen=109 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=664 | text_est=466 | nontext_est=198 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=80% | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** title words=3
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2335 | text_est=466 | nontext_est=1869 | gen=84 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1763 | text_est=466 | nontext_est=1297 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=3.90% | nontext prompt burden=85% | missing terms: Bird, Holiday, People, Riverbank, Stone wall | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=caveat | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3152 | text_est=466 | nontext_est=2686 | gen=123 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 85% and the output stays weak under that load.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1763 | text_est=466 | nontext_est=1297 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday | nonvisual metadata reused
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=811 | text_est=466 | nontext_est=345 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=552 | text_est=466 | nontext_est=86 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=70% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1544 | text_est=466 | nontext_est=1078 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-4-31b-it-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: Berkshire, Bird, Holiday, Passenger, People | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=825 | text_est=466 | nontext_est=359 | gen=97 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=3.65% | nontext prompt burden=86% | missing terms: Bird, Blue sky | keywords=19
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=4 | keywords=19
- **Utility:** user=caveat | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3342 | text_est=466 | nontext_est=2876 | gen=122 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 86% and the output stays weak under that load.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Berkshire, Bird, Holiday, Lifebuoy, People | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=819 | text_est=466 | nontext_est=353 | gen=252 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=70% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1544 | text_est=466 | nontext_est=1078 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=820 | text_est=466 | nontext_est=354 | gen=88 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 61 occurrences). | nontext prompt burden=82% | missing sections: description, keywords | missing terms: Berkshire, Bird, Holiday, Lifebuoy, Passenger
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=63
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2654 | text_est=466 | nontext_est=2188 | gen=107 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=70% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1544 | text_est=466 | nontext_est=1078 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|end|&gt; appeared in generated text. | Special control token &lt;|endoftext|&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=66%
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=47
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1382 | text_est=466 | nontext_est=916 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=93% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Holiday, People, Quay
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6667 | text_est=466 | nontext_est=6201 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=86% | missing sections: title, keywords | missing terms: Berkshire, Bird, Holiday, Lifebuoy, People
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, keywords | description sentences=7
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3433 | text_est=466 | nontext_est=2967 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | nontext prompt burden=75% | missing terms: Berkshire, Bird, Holiday, Passenger, People | keywords=30
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=30
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1873 | text_est=466 | nontext_est=1407 | gen=414 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=90% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4646 | text_est=466 | nontext_est=4180 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=93% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Holiday, Lifebuoy, Sightseeing
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6667 | text_est=466 | nontext_est=6201 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=84% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2834 | text_est=466 | nontext_est=2368 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=73% | missing terms: Bird, Blue sky, filled, cruises, along | keywords=19 | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=19
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1731 | text_est=466 | nontext_est=1265 | gen=250 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** context_budget | user=caveat
- **Why:** At long prompt length (16780 tokens), output may stop following prompt/image context. | output/prompt=0.44% | nontext prompt burden=97% | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** title words=3
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16780 | text_est=466 | nontext_est=16314 | gen=73 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** At long prompt length (16769 tokens), output became repetitive. | hit token cap (500) | nontext prompt burden=97% | keyword duplication=86%
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=4 | keywords=170 | keyword duplication=0.86
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16769 | text_est=466 | nontext_est=16303 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect long-context cache behavior under heavy image-token burden.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** degrades trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | degrades trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16771 | text_est=466 | nontext_est=16305 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-4-31b-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy | nonvisual metadata reused
- **Trusted hints:** degrades trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | degrades trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=813 | text_est=466 | nontext_est=347 | gen=100 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=86% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3342 | text_est=466 | nontext_est=2876 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Holiday, Lifebuoy | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=523 | text_est=466 | nontext_est=57 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=80% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2335 | text_est=466 | nontext_est=1869 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=93% | missing sections: title, description | missing terms: Berkshire, Bird, Holiday, People, Sightseeing
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description | keywords=42
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6667 | text_est=466 | nontext_est=6201 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=73% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday | repetitive token=phrase: "\*/ \*/ \*/ \*/..."
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1731 | text_est=466 | nontext_est=1265 | gen=419 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: description, keywords | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=16
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16794 | text_est=466 | nontext_est=16328 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|endoftext|&gt; appeared in generated text. | Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=16780 | text_est=466 | nontext_est=16314 | gen=12 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16794 | text_est=466 | nontext_est=16328 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: filled, passengers, cruises, along, background
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16794 | text_est=466 | nontext_est=16328 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** At long prompt length (16794 tokens), output may stop following prompt/image context. | hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Berkshire, Bird, Blue sky, Castle, Holiday
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16794 | text_est=466 | nontext_est=16328 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect long-context cache behavior under heavy image-token burden.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Holiday, Passenger
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16794 | text_est=466 | nontext_est=16328 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: Holiday, Lifebuoy, Sightseeing, Tree, filled
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16794 | text_est=466 | nontext_est=16328 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: Berkshire, Bird, Blue sky, Holiday, Sightseeing | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=524 | text_est=466 | nontext_est=58 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.
