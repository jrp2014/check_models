# Automated Review Digest

_Generated on 2026-04-19 02:03:59 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (89/100) | Desc 93 | Keywords 77 | Δ+15 | 54.1 tps

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | Δ-68 | 20.6 tps | harness, metadata borrowing, missing sections
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (6/100) | Desc 23 | Keywords 42 | Δ-68 | 22.5 tps | context ignored, harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (19/100) | Desc 46 | Keywords 42 | Δ-55 | 5.7 tps | context ignored, harness, missing sections
- `microsoft/Phi-3.5-vision-instruct`: 🟡 C (60/100) | Desc 96 | Keywords 54 | Δ-14 | 50.2 tps | degeneration, harness, metadata borrowing
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) | Desc 60 | Keywords 0 | Δ-74 | 125.5 tps | context ignored, cutoff, missing sections

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict   | Hint Handling           | Key Evidence                                                                              |
|-----------------------------------------------------|-----------|-------------------------|-------------------------------------------------------------------------------------------|
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`   | preserves trusted hints | nontext prompt burden=86% \| missing terms: Activities, Berkshire, Couple, Door, Fortress |

### `caveat`

| Model                                               | Verdict          | Hint Handling                                                                       | Key Evidence                                                                                                                                  |
|-----------------------------------------------------|------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `context_budget` | preserves trusted hints \| nonvisual metadata reused                                | output/prompt=3.72% \| nontext prompt burden=86% \| missing terms: Activities, Berkshire, Couple, Door, Fortress \| nonvisual metadata reused |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`          | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door | nontext prompt burden=81% \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                       |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `context_budget` | preserves trusted hints \| nonvisual metadata reused                                | output/prompt=3.88% \| nontext prompt burden=86% \| missing terms: Activities, Couple, Door, Kissing, Man \| nonvisual metadata reused        |
| `mlx-community/pixtral-12b-8bit`                    | `context_budget` | preserves trusted hints \| nonvisual metadata reused                                | output/prompt=3.18% \| nontext prompt burden=87% \| missing terms: Activities, Berkshire, Kissing, Man, Standing \| keywords=21               |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                                    | Key Evidence                                                                                                                                                                                                                                                |
|---------------------------------------------------------|---------------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Qwen/Qwen3-VL-2B-Instruct`                             | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                             | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=55 \| context echo=100% \| nonvisual metadata reused                                                                                                                       |
| `ggml-org/gemma-3-1b-it-GGUF`                           | `runtime_failure`   | not evaluated                                                                                                    | model error \| model config model load model                                                                                                                                                                                                                |
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure`   | not evaluated                                                                                                    | processor error \| model config processor load processor                                                                                                                                                                                                    |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                             | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=55 \| context echo=100% \| nonvisual metadata reused                                                                                                                       |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                             | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=56 \| context echo=100% \| nonvisual metadata reused                                                                                                                       |
| `mlx-community/X-Reasoner-7B-8bit`                      | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                             | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=55 \| context echo=100% \| nonvisual metadata reused                                                                                                                       |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | degrades trusted hints                                                                                           | missing sections: keywords \| missing terms: Activities, Berkshire, Door, Fortress, Kissing                                                                                                                                                                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Door, Fortress, Man \| nonvisual metadata reused \| reasoning leak                                                                                                  |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `harness`           | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | Output appears truncated to about 6 tokens. \| nontext prompt burden=71% \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                                                      |
| `mlx-community/FastVLM-0.5B-bf16`                       | `model_shortcoming` | ignores trusted hints \| missing terms: Activities, Berkshire, Couple, Door, Fortress                            | missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress                                                                                                                                              |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| missing terms: Activities, Berkshire, Door, Fortress, Kissing \| keyword duplication=89% \| nonvisual metadata reused                                                                                                                |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | nontext prompt burden=67% \| missing terms: Activities, Berkshire, Door, Fortress, Man \| keywords=20 \| nonvisual metadata reused                                                                                                                          |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `harness`           | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `model_shortcoming` | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | nontext prompt burden=84% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                                                   |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                                                         |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                            |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door \| nonvisual metadata reused                                                                                            |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                            |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing terms: Berkshire, Pedestrians, Standing, towering, over \| nonvisual metadata reused                                                                                                                                                                |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                            |
| `mlx-community/gemma-4-31b-it-4bit`                     | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing terms: Activities, Berkshire, Couple, Door, Fortress \| nonvisual metadata reused                                                                                                                                                                   |
| `qnguyen3/nanoLLaVA`                                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door \| repetitive token=phrase: "motorcycle painting glasses, m..."                                                         |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress \| nonvisual metadata reused                                                                                                                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                            |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                             | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 53 occurrences). \| nontext prompt burden=83% \| missing sections: description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress                         |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                            |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing terms: Berkshire, towering, over, stand, Royal \| nonvisual metadata reused                                                                                                                                                                         |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                             | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=67%                                                            |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                            |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing sections: title, description, keywords \| missing terms: Berkshire, Pedestrians, stand \| nonvisual metadata reused                                                                                                                                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | hit token cap (500) \| nontext prompt burden=84% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                            |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=87% \| missing sections: description, keywords \| missing terms: Activities, Berkshire, Door, Fortress, Kissing                                                                                                |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Door, Fortress, Kissing                                                                                         |
| `mlx-community/InternVL3-8B-bf16`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | hit token cap (500) \| nontext prompt burden=81% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                            |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                            |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `cutoff_degraded`   | degrades trusted hints                                                                                           | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress                                                                                          |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=76% \| missing sections: title, description \| missing terms: Activities, Berkshire, Door, Kissing, Man                                                                                                        |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress                                                                                          |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress \| nonvisual metadata reused                                                                                                                 |
| `mlx-community/GLM-4.6V-nvfp4`                          | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: Activities, Berkshire, Door, Fortress, Kissing                                                                                                   |
| `mlx-community/pixtral-12b-bf16`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Couple, Door, Fortress                            | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress                                                                                          |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=97% \| missing sections: keywords \| nonvisual metadata reused                                                                                                                                                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Activities, Berkshire, Castle, Couple, Door                              | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                            |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | degrades trusted hints                                                                                           | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Door, Fortress, Kissing                                                                                         |
| `mlx-community/gemma-4-31b-bf16`                        | `cutoff_degraded`   | degrades trusted hints \| nonvisual metadata reused                                                              | hit token cap (500) \| missing sections: title \| missing terms: Activities, Berkshire, Couple, Door, Kissing \| keywords=64                                                                                                                                |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress                                                                                          |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=97% \| missing sections: description, keywords \| missing terms: towering, over, Below, stand, pavement                                                                                                        |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=97% \| missing sections: description, keywords \| nonvisual metadata reused                                                                                                                                    |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| nonvisual metadata reused                                                                                                                             |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `mlx-vlm`

| Model                                                   | Verdict   | Evidence                                                                                                                                                                                                                            | Next Action                                                                         |
|---------------------------------------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `Qwen/Qwen3-VL-2B-Instruct`                             | `harness` | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=55 \| context echo=100% \| nonvisual metadata reused                                                                                               | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness` | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=55 \| context echo=100% \| nonvisual metadata reused                                                                                               | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | `harness` | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=56 \| context echo=100% \| nonvisual metadata reused                                                                                               | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/X-Reasoner-7B-8bit`                      | `harness` | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=55 \| context echo=100% \| nonvisual metadata reused                                                                                               | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 53 occurrences). \| nontext prompt burden=83% \| missing sections: description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress | Inspect decode cleanup; tokenizer markers are leaking into user-facing text.        |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=67%                                    | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model                                               | Verdict             | Evidence                                                                                                                                                                                            | Next Action                                                                                                        |
|-----------------------------------------------------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `model_shortcoming` | missing sections: keywords \| missing terms: Activities, Berkshire, Door, Fortress, Kissing                                                                                                         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/LFM2-VL-1.6B-8bit`                   | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Door, Fortress, Man \| nonvisual metadata reused \| reasoning leak                                          | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/FastVLM-0.5B-bf16`                   | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress                                                                                      | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `context_budget`    | output/prompt=3.72% \| nontext prompt burden=86% \| missing terms: Activities, Berkshire, Couple, Door, Fortress \| nonvisual metadata reused                                                       | Treat this as a prompt-budget issue first; nontext prompt burden is 86% and the output stays weak under that load. |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                 | `cutoff_degraded`   | hit token cap (500) \| missing terms: Activities, Berkshire, Door, Fortress, Kissing \| keyword duplication=89% \| nonvisual metadata reused                                                        | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        | `model_shortcoming` | nontext prompt burden=67% \| missing terms: Activities, Berkshire, Door, Fortress, Man \| keywords=20 \| nonvisual metadata reused                                                                  | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/llava-v1.6-mistral-7b-8bit`          | `model_shortcoming` | nontext prompt burden=84% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`          | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                 | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-3n-E2B-4bit`                   | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door \| nonvisual metadata reused                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/SmolVLM-Instruct-bf16`               | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `model_shortcoming` | missing terms: Berkshire, Pedestrians, Standing, towering, over \| nonvisual metadata reused                                                                                                        | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-4-31b-it-4bit`                 | `model_shortcoming` | missing terms: Activities, Berkshire, Couple, Door, Fortress \| nonvisual metadata reused                                                                                                           | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `qnguyen3/nanoLLaVA`                                | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door \| repetitive token=phrase: "motorcycle painting glasses, m..." | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`             | nontext prompt burden=86% \| missing terms: Activities, Berkshire, Couple, Door, Fortress                                                                                                           | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`             | nontext prompt burden=81% \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                                             | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `context_budget`    | output/prompt=3.88% \| nontext prompt burden=86% \| missing terms: Activities, Couple, Door, Kissing, Man \| nonvisual metadata reused                                                              | Treat this as a prompt-budget issue first; nontext prompt burden is 86% and the output stays weak under that load. |
| `mlx-community/pixtral-12b-8bit`                    | `context_budget`    | output/prompt=3.18% \| nontext prompt burden=87% \| missing terms: Activities, Berkshire, Kissing, Man, Standing \| keywords=21                                                                     | Treat this as a prompt-budget issue first; nontext prompt burden is 87% and the output stays weak under that load. |
| `mlx-community/gemma-3n-E4B-it-bf16`                | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress \| nonvisual metadata reused                                                         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`           | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `model_shortcoming` | missing terms: Berkshire, towering, over, stand, Royal \| nonvisual metadata reused                                                                                                                 | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/paligemma2-3b-pt-896-4bit`           | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Berkshire, Pedestrians, stand \| nonvisual metadata reused                                                                         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=84% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`     | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=87% \| missing sections: description, keywords \| missing terms: Activities, Berkshire, Door, Fortress, Kissing                                        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Door, Fortress, Kissing                                 | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/InternVL3-8B-bf16`                   | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=81% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Molmo-7B-D-0924-8bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress                                  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`  | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=76% \| missing sections: title, description \| missing terms: Activities, Berkshire, Door, Kissing, Man                                                | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-Flash-6bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress                                  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress \| nonvisual metadata reused                                                         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-nvfp4`                      | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: Activities, Berkshire, Door, Fortress, Kissing                                           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/pixtral-12b-bf16`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress                                  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: keywords \| nonvisual metadata reused                                                                                         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Molmo-7B-D-0924-bf16`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Door, Fortress, Kissing                                 | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-4-31b-bf16`                    | `cutoff_degraded`   | hit token cap (500) \| missing sections: title \| missing terms: Activities, Berkshire, Couple, Door, Kissing \| keywords=64                                                                        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Couple, Door, Fortress                                  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: description, keywords \| missing terms: towering, over, Below, stand, pavement                                                | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: description, keywords \| nonvisual metadata reused                                                                            | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-4bit`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| nonvisual metadata reused                                                                     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |

### `model-config`

| Model                                            | Verdict           | Evidence                                                                                                                                                                                                                                                    | Next Action                                                                                          |
|--------------------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `ggml-org/gemma-3-1b-it-GGUF`                    | `runtime_failure` | model error \| model config model load model                                                                                                                                                                                                                | Inspect model repo config, chat template, and EOS settings.                                          |
| `mlx-community/MolmoPoint-8B-fp16`               | `runtime_failure` | processor error \| model config processor load processor                                                                                                                                                                                                    | Inspect model repo config, chat template, and EOS settings.                                          |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`  | `harness`         | Output appears truncated to about 6 tokens. \| nontext prompt burden=71% \| missing terms: Activities, Berkshire, Castle, Couple, Door                                                                                                                      | Inspect model repo config, chat template, and EOS settings.                                          |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness`         | Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Activities, Berkshire, Castle, Couple, Door | Check chat-template and EOS defaults first; the output shape is not matching the requested contract. |

## Model Verdicts

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|im_end|&gt; appeared in generated text. | keywords=55 | context echo=100% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=55
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=449 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

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

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|im_end|&gt; appeared in generated text. | keywords=55 | context echo=100% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=55
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=449 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|im_end|&gt; appeared in generated text. | keywords=56 | context echo=100% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=56
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=449 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|im_end|&gt; appeared in generated text. | keywords=55 | context echo=100% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=55
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=449 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: keywords | missing terms: Activities, Berkshire, Door, Fortress, Kissing
- **Trusted hints:** degrades trusted hints
- **Contract:** missing: keywords | title words=11
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=530 | text_est=449 | nontext_est=81 | gen=46 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Activities, Berkshire, Door, Fortress, Man | nonvisual metadata reused | reasoning leak
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=790 | text_est=449 | nontext_est=341 | gen=184 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output appears truncated to about 6 tokens. | nontext prompt burden=71% | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1558 | text_est=449 | nontext_est=1109 | gen=6 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Activities, Berkshire, Couple, Door, Fortress
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Couple, Door, Fortress
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=534 | text_est=449 | nontext_est=85 | gen=32 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=3.72% | nontext prompt burden=86% | missing terms: Activities, Berkshire, Couple, Door, Fortress | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=caveat | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3143 | text_est=449 | nontext_est=2694 | gen=117 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 86% and the output stays weak under that load.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing terms: Activities, Berkshire, Door, Fortress, Kissing | keyword duplication=89% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** description sentences=5 | keywords=172 | keyword duplication=0.89
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=600 | text_est=449 | nontext_est=151 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=67% | missing terms: Activities, Berkshire, Door, Fortress, Man | keywords=20 | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=4 | keywords=20
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1363 | text_est=449 | nontext_est=914 | gen=167 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=71% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1558 | text_est=449 | nontext_est=1109 | gen=12 | max=500 | stop=completed
- **Next action:** Check chat-template and EOS defaults first; the output shape is not matching the requested contract.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=84% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2761 | text_est=449 | nontext_est=2312 | gen=17 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=645 | text_est=449 | nontext_est=196 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1745 | text_est=449 | nontext_est=1296 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door | nonvisual metadata reused
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=793 | text_est=449 | nontext_est=344 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1745 | text_est=449 | nontext_est=1296 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: Berkshire, Pedestrians, Standing, towering, over | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=802 | text_est=449 | nontext_est=353 | gen=91 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=71% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1525 | text_est=449 | nontext_est=1076 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-4-31b-it-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: Activities, Berkshire, Couple, Door, Fortress | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=807 | text_est=449 | nontext_est=358 | gen=82 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door | repetitive token=phrase: "motorcycle painting glasses, m..."
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=530 | text_est=449 | nontext_est=81 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=86% | missing terms: Activities, Berkshire, Couple, Door, Fortress
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3144 | text_est=449 | nontext_est=2695 | gen=105 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=81% | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** title words=2
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2313 | text_est=449 | nontext_est=1864 | gen=81 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=3.88% | nontext prompt burden=86% | missing terms: Activities, Couple, Door, Kissing, Man | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=caveat | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3144 | text_est=449 | nontext_est=2695 | gen=122 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 86% and the output stays weak under that load.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=3.18% | nontext prompt burden=87% | missing terms: Activities, Berkshire, Kissing, Man, Standing | keywords=21
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=21
- **Utility:** user=caveat | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3337 | text_est=449 | nontext_est=2888 | gen=106 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 87% and the output stays weak under that load.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Activities, Berkshire, Couple, Door, Fortress | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=801 | text_est=449 | nontext_est=352 | gen=348 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=71% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1525 | text_est=449 | nontext_est=1076 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 53 occurrences). | nontext prompt burden=83% | missing sections: description, keywords | missing terms: Activities, Berkshire, Couple, Door, Fortress
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=55
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2646 | text_est=449 | nontext_est=2197 | gen=82 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=71% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1525 | text_est=449 | nontext_est=1076 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: Berkshire, towering, over, stand, Royal | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=802 | text_est=449 | nontext_est=353 | gen=96 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|end|&gt; appeared in generated text. | Special control token &lt;|endoftext|&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=67%
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=4 | keywords=50
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1363 | text_est=449 | nontext_est=914 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=90% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4630 | text_est=449 | nontext_est=4181 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Berkshire, Pedestrians, stand | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=501 | text_est=449 | nontext_est=52 | gen=126 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=84% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2813 | text_est=449 | nontext_est=2364 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=87% | missing sections: description, keywords | missing terms: Activities, Berkshire, Door, Fortress, Kissing
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=131
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3428 | text_est=449 | nontext_est=2979 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=93% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Door, Fortress, Kissing
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6645 | text_est=449 | nontext_est=6196 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=81% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2313 | text_est=449 | nontext_est=1864 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1717 | text_est=449 | nontext_est=1268 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=71% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Couple, Door, Fortress
- **Trusted hints:** degrades trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1558 | text_est=449 | nontext_est=1109 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=76% | missing sections: title, description | missing terms: Activities, Berkshire, Door, Kissing, Man
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description | keywords=43
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1852 | text_est=449 | nontext_est=1403 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=93% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Couple, Door, Fortress
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6645 | text_est=449 | nontext_est=6196 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Activities, Berkshire, Couple, Door, Fortress | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=502 | text_est=449 | nontext_est=53 | gen=167 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=93% | missing sections: title, description | missing terms: Activities, Berkshire, Door, Fortress, Kissing
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description | keywords=23
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6645 | text_est=449 | nontext_est=6196 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=87% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Couple, Door, Fortress
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Couple, Door, Fortress
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3337 | text_est=449 | nontext_est=2888 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: keywords | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords | title words=57
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16773 | text_est=449 | nontext_est=16324 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Trusted hints:** ignores trusted hints | missing terms: Activities, Berkshire, Castle, Couple, Door
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1717 | text_est=449 | nontext_est=1268 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Door, Fortress, Kissing
- **Trusted hints:** degrades trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16773 | text_est=449 | nontext_est=16324 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-4-31b-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title | missing terms: Activities, Berkshire, Couple, Door, Kissing | keywords=64
- **Trusted hints:** degrades trusted hints | nonvisual metadata reused
- **Contract:** missing: title | keywords=64
- **Utility:** user=avoid | degrades trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=795 | text_est=449 | nontext_est=346 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: Activities, Berkshire, Couple, Door, Fortress
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16773 | text_est=449 | nontext_est=16324 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: description, keywords | missing terms: towering, over, Below, stand, pavement
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=25
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16773 | text_est=449 | nontext_est=16324 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: description, keywords | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=17
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16773 | text_est=449 | nontext_est=16324 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16773 | text_est=449 | nontext_est=16324 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.
