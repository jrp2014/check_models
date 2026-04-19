# Automated Review Digest

_Generated on 2026-04-19 23:34:45 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (96/100) | Desc 100 | Keywords 77 | Δ+50 | 115.2 tps
- `mlx-community/gemma-4-31b-it-4bit`: 🏆 A (94/100) | Desc 100 | Keywords 83 | Δ+49 | 27.5 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (92/100) | Desc 100 | Keywords 91 | Δ+47 | 66.3 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (90/100) | Desc 93 | Keywords 91 | Δ+45 | 63.2 tps

### Watchlist

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) | Desc 21 | Keywords 0 | Δ-40 | 370.3 tps | context ignored, harness
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (6/100) | Desc 45 | Keywords 0 | Δ-39 | 70.7 tps | context ignored, harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | Δ-39 | 32.0 tps | harness, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (9/100) | Desc 45 | Keywords 32 | Δ-36 | 5.9 tps | context ignored, harness
- `microsoft/Phi-3.5-vision-instruct`: 🟡 C (58/100) | Desc 60 | Keywords 48 | Δ+12 | 56.4 tps | hallucination, harness, metadata borrowing

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict     | Hint Handling                                                               | Key Evidence                                                                                                                                                                                    |
|-----------------------------------------------------|-------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`     | improves trusted hints                                                      | missing terms: view, Round, Windsor, royal, residence                                                                                                                                           |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`     | improves trusted hints                                                      | nontext prompt burden=86% \| missing terms: royal, residence, Berkshire, seen, across                                                                                                           |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`     | improves trusted hints                                                      | nontext prompt burden=86% \| missing terms: seen, which, indicates, reigning, monarch                                                                                                           |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`     | improves trusted hints                                                      | missing terms: view, Round, Windsor, royal, residence                                                                                                                                           |
| `mlx-community/X-Reasoner-7B-8bit`                  | `token_cap` | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle | At long prompt length (16731 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=97% \| missing terms: view, Round, Tower, Windsor, Castle |

### `caveat`

| Model                                             | Verdict          | Hint Handling                                                                | Key Evidence                                                                                                                              |
|---------------------------------------------------|------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `context_budget` | improves trusted hints \| nonvisual metadata reused                          | output/prompt=3.56% \| nontext prompt burden=86% \| missing terms: seen, across, flying, indicates, reigning \| nonvisual metadata reused |
| `mlx-community/InternVL3-14B-8bit`                | `clean`          | ignores trusted hints \| missing terms: Round, Tower, Windsor, Castle, royal | nontext prompt burden=81% \| missing terms: Round, Tower, Windsor, Castle, royal                                                          |
| `mlx-community/pixtral-12b-8bit`                  | `context_budget` | improves trusted hints \| nonvisual metadata reused                          | output/prompt=3.02% \| nontext prompt burden=87% \| missing terms: view, royal, seen, across, flying \| nonvisual metadata reused         |
| `Qwen/Qwen3-VL-2B-Instruct`                       | `context_budget` | preserves trusted hints \| nonvisual metadata reused                         | output/prompt=0.99% \| nontext prompt burden=97% \| nonvisual metadata reused                                                             |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                            | Key Evidence                                                                                                                                                                                                                   |
|---------------------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ggml-org/gemma-3-1b-it-GGUF`                           | `runtime_failure`   | not evaluated                                                                                            | model error \| model config model load model                                                                                                                                                                                   |
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure`   | not evaluated                                                                                            | processor error \| model config processor load processor                                                                                                                                                                       |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | degrades trusted hints \| nonvisual metadata reused                                                      | missing sections: keywords \| missing terms: view, Round, residence, Berkshire, seen \| nonvisual metadata reused                                                                                                              |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | missing sections: title, description, keywords \| context echo=41% \| nonvisual metadata reused                                                                                                                                |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `model_shortcoming` | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                                              |
| `mlx-community/FastVLM-0.5B-bf16`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, royal \| nonvisual metadata reused  | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, royal \| nonvisual metadata reused                                                                        |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `harness`           | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | Output was a short generic filler response (about 8 tokens). \| nontext prompt burden=84% \| missing terms: view, Round, Tower, Windsor, Castle                                                                                |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                     | hit token cap (500) \| keyword duplication=75% \| nonvisual metadata reused \| repetitive token=phrase: "2026-04-18, 17:45:40 bst, 2026..."                                                                                    |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `harness`           | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=71% \| missing terms: view, Round, Tower, Windsor, Castle                              |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | nontext prompt burden=66% \| context echo=40% \| nonvisual metadata reused                                                                                                                                                     |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                      | missing terms: royal, residence, seen, which, indicates \| nonvisual metadata reused                                                                                                                                           |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                                                    |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle \| nonvisual metadata reused                                                                       |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                       |
| `qnguyen3/nanoLLaVA`                                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle \| repetitive token=phrase: "painting glasses glasses, moto..."                                    |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                       |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                       |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                       |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | preserves trusted hints                                                                                  | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 57 occurrences). \| nontext prompt burden=83% \| missing sections: description, keywords \| missing terms: view, royal, residence, Berkshire, which |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                      | missing terms: royal, residence, which, indicates, reigning \| nonvisual metadata reused                                                                                                                                       |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                      | missing terms: Round, royal, residence, Berkshire, seen \| keywords=24 \| nonvisual metadata reused                                                                                                                            |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                       |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | missing sections: title, description, keywords \| missing terms: which, indicates \| nonvisual metadata reused                                                                                                                 |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                     | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=66%                               |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: royal, residence, Berkshire, which, indicates                                                             |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                       |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: view, royal, residence, Berkshire, seen                                                                   |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: view, royal, residence, Berkshire, seen                                                                             |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | hit token cap (500) \| nontext prompt burden=84% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                       |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                     | hit token cap (500) \| nontext prompt burden=76% \| missing sections: title, description, keywords \| missing terms: photograph                                                                                                |
| `mlx-community/InternVL3-8B-bf16`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | hit token cap (500) \| nontext prompt burden=81% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                       |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | At long prompt length (16722 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                                          |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Tower, Windsor, Castle, royal                              | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: view, Tower, Windsor, Castle, royal                                                                       |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `cutoff_degraded`   | degrades trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, royal, residence                                                                      |
| `mlx-community/GLM-4.6V-nvfp4`                          | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: view, royal, residence, Berkshire, seen                                                                             |
| `mlx-community/pixtral-12b-bf16`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, royal                               | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, royal                                                                        |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: royal, Berkshire, photograph                                                                              |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| context echo=100%                                                                                                        |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=97% \| missing sections: description, keywords \| missing terms: royal, residence, Berkshire, seen, across                                                                        |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                       |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness`           | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| Output appears truncated to about 2 tokens. \| nontext prompt burden=97% \| missing terms: view, Round, Tower, Windsor, Castle                      |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: royal, residence, Berkshire, flagpole, which                                                              |
| `mlx-community/gemma-4-31b-bf16`                        | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, seen, across, flying, flagpole \| nonvisual metadata reused                                                                      |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=97% \| missing sections: description, keywords \| missing terms: royal, Berkshire, seen, across, Thames                                                                           |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: view, Berkshire, seen, across, River                                                                      |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                     | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: residence, Berkshire, Union, indicates, reigning \| nonvisual metadata reused                                                          |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `mlx`

| Model                                     | Verdict           | Evidence                                                                                                                                                                                        | Next Action                                                         |
|-------------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff_degraded` | At long prompt length (16722 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                           | Inspect long-context cache behavior under heavy image-token burden. |
| `mlx-community/X-Reasoner-7B-8bit`        | `token_cap`       | At long prompt length (16731 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=97% \| missing terms: view, Round, Tower, Windsor, Castle | Inspect long-context cache behavior under heavy image-token burden. |

### `mlx-vlm`

| Model                                                   | Verdict   | Evidence                                                                                                                                                                                                                       | Next Action                                                                         |
|---------------------------------------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 57 occurrences). \| nontext prompt burden=83% \| missing sections: description, keywords \| missing terms: view, royal, residence, Berkshire, which | Inspect decode cleanup; tokenizer markers are leaking into user-facing text.        |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=66%                               | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness` | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| Output appears truncated to about 2 tokens. \| nontext prompt burden=97% \| missing terms: view, Round, Tower, Windsor, Castle                      | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model                                               | Verdict             | Evidence                                                                                                                                                                                    | Next Action                                                                                                        |
|-----------------------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `model_shortcoming` | missing sections: keywords \| missing terms: view, Round, residence, Berkshire, seen \| nonvisual metadata reused                                                                           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/LFM2-VL-1.6B-8bit`                   | `model_shortcoming` | missing sections: title, description, keywords \| context echo=41% \| nonvisual metadata reused                                                                                             | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`             | missing terms: view, Round, Windsor, royal, residence                                                                                                                                       | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `context_budget`    | output/prompt=3.56% \| nontext prompt burden=86% \| missing terms: seen, across, flying, indicates, reigning \| nonvisual metadata reused                                                   | Treat this as a prompt-budget issue first; nontext prompt burden is 86% and the output stays weak under that load. |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | `model_shortcoming` | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/FastVLM-0.5B-bf16`                   | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, royal \| nonvisual metadata reused                                     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                 | `cutoff_degraded`   | hit token cap (500) \| keyword duplication=75% \| nonvisual metadata reused \| repetitive token=phrase: "2026-04-18, 17:45:40 bst, 2026..."                                                 | Treat as a model-quality limitation for this prompt and image.                                                     |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        | `model_shortcoming` | nontext prompt burden=66% \| context echo=40% \| nonvisual metadata reused                                                                                                                  | Treat as a model-quality limitation for this prompt and image.                                                     |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`             | nontext prompt burden=86% \| missing terms: royal, residence, Berkshire, seen, across                                                                                                       | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `model_shortcoming` | missing terms: royal, residence, seen, which, indicates \| nonvisual metadata reused                                                                                                        | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`             | nontext prompt burden=81% \| missing terms: Round, Tower, Windsor, Castle, royal                                                                                                            | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`             | nontext prompt burden=86% \| missing terms: seen, which, indicates, reigning, monarch                                                                                                       | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`          | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                 | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-3n-E2B-4bit`                   | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle \| nonvisual metadata reused                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`             | missing terms: view, Round, Windsor, royal, residence                                                                                                                                       | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/pixtral-12b-8bit`                    | `context_budget`    | output/prompt=3.02% \| nontext prompt burden=87% \| missing terms: view, royal, seen, across, flying \| nonvisual metadata reused                                                           | Treat this as a prompt-budget issue first; nontext prompt burden is 87% and the output stays weak under that load. |
| `qnguyen3/nanoLLaVA`                                | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle \| repetitive token=phrase: "painting glasses glasses, moto..." | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/SmolVLM-Instruct-bf16`               | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`           | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `model_shortcoming` | missing terms: royal, residence, which, indicates, reigning \| nonvisual metadata reused                                                                                                    | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/gemma-3n-E4B-it-bf16`                | `model_shortcoming` | missing terms: Round, royal, residence, Berkshire, seen \| keywords=24 \| nonvisual metadata reused                                                                                         | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: which, indicates \| nonvisual metadata reused                                                                              | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: royal, residence, Berkshire, which, indicates                          | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/paligemma2-3b-pt-896-4bit`           | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`     | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: view, royal, residence, Berkshire, seen                                | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-Flash-6bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: view, royal, residence, Berkshire, seen                                          | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `Qwen/Qwen3-VL-2B-Instruct`                         | `context_budget`    | output/prompt=0.99% \| nontext prompt burden=97% \| nonvisual metadata reused                                                                                                               | Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load. |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=84% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`  | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=76% \| missing sections: title, description, keywords \| missing terms: photograph                                                             | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/InternVL3-8B-bf16`                   | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=81% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`     | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: view, Tower, Windsor, Castle, royal                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Molmo-7B-D-0924-8bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, royal, residence                                   | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-nvfp4`                      | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: view, royal, residence, Berkshire, seen                                          | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/pixtral-12b-bf16`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, royal                                     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: royal, Berkshire, photograph                                           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| context echo=100%                                                                     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: description, keywords \| missing terms: royal, residence, Berkshire, seen, across                                     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Molmo-7B-D-0924-bf16`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: royal, residence, Berkshire, flagpole, which                           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-4-31b-bf16`                    | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, seen, across, flying, flagpole \| nonvisual metadata reused                                   | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-4bit`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: description, keywords \| missing terms: royal, Berkshire, seen, across, Thames                                        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: view, Berkshire, seen, across, River                                   | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: residence, Berkshire, Union, indicates, reigning \| nonvisual metadata reused                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |

### `model-config`

| Model                                            | Verdict           | Evidence                                                                                                                                                                                          | Next Action                                                 |
|--------------------------------------------------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `ggml-org/gemma-3-1b-it-GGUF`                    | `runtime_failure` | model error \| model config model load model                                                                                                                                                      | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/MolmoPoint-8B-fp16`               | `runtime_failure` | processor error \| model config processor load processor                                                                                                                                          | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/llava-v1.6-mistral-7b-8bit`       | `harness`         | Output was a short generic filler response (about 8 tokens). \| nontext prompt burden=84% \| missing terms: view, Round, Tower, Windsor, Castle                                                   | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness`         | Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=71% \| missing terms: view, Round, Tower, Windsor, Castle | Inspect model repo config, chat template, and EOS settings. |

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
- **Why:** missing sections: keywords | missing terms: view, Round, residence, Berkshire, seen | nonvisual metadata reused
- **Trusted hints:** degrades trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords | title words=11
- **Utility:** user=avoid | degrades trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=503 | text_est=442 | nontext_est=61 | gen=89 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | context echo=41% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=758 | text_est=442 | nontext_est=316 | gen=117 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-4-26b-a4b-it-4bit`

- **Verdict:** clean | user=recommended
- **Why:** missing terms: view, Round, Windsor, royal, residence
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=779 | text_est=442 | nontext_est=337 | gen=76 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=3.56% | nontext prompt burden=86% | missing terms: seen, across, flying, indicates, reigning | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=caveat | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3114 | text_est=442 | nontext_est=2672 | gen=111 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 86% and the output stays weak under that load.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=71% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1531 | text_est=442 | nontext_est=1089 | gen=24 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, royal | nonvisual metadata reused
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, royal | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=507 | text_est=442 | nontext_est=65 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** harness | user=avoid
- **Why:** Output was a short generic filler response (about 8 tokens). | nontext prompt burden=84% | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=2722 | text_est=442 | nontext_est=2280 | gen=8 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | keyword duplication=75% | nonvisual metadata reused | repetitive token=phrase: "2026-04-18, 17:45:40 bst, 2026..."
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=12 | keywords=55 | keyword duplication=0.75
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=568 | text_est=442 | nontext_est=126 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=71% | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1531 | text_est=442 | nontext_est=1089 | gen=8 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=66% | context echo=40% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1319 | text_est=442 | nontext_est=877 | gen=164 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=86% | missing terms: royal, residence, Berkshire, seen, across
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3115 | text_est=442 | nontext_est=2673 | gen=97 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: royal, residence, seen, which, indicates | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=774 | text_est=442 | nontext_est=332 | gen=82 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=81% | missing terms: Round, Tower, Windsor, Castle, royal
- **Trusted hints:** ignores trusted hints | missing terms: Round, Tower, Windsor, Castle, royal
- **Contract:** title words=2
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2286 | text_est=442 | nontext_est=1844 | gen=71 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=86% | missing terms: seen, which, indicates, reigning, monarch
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3115 | text_est=442 | nontext_est=2673 | gen=111 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=616 | text_est=442 | nontext_est=174 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle | nonvisual metadata reused
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=765 | text_est=442 | nontext_est=323 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-4-31b-it-4bit`

- **Verdict:** clean | user=recommended
- **Why:** missing terms: view, Round, Windsor, royal, residence
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=779 | text_est=442 | nontext_est=337 | gen=84 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1716 | text_est=442 | nontext_est=1274 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=3.02% | nontext prompt burden=87% | missing terms: view, royal, seen, across, flying | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=caveat | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3308 | text_est=442 | nontext_est=2866 | gen=100 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 87% and the output stays weak under that load.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle | repetitive token=phrase: "painting glasses glasses, moto..."
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=503 | text_est=442 | nontext_est=61 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1716 | text_est=442 | nontext_est=1274 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=70% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1497 | text_est=442 | nontext_est=1055 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=70% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1497 | text_est=442 | nontext_est=1055 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 57 occurrences). | nontext prompt burden=83% | missing sections: description, keywords | missing terms: view, royal, residence, Berkshire, which
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: description, keywords | title words=59
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2617 | text_est=442 | nontext_est=2175 | gen=81 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: royal, residence, which, indicates, reigning | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=774 | text_est=442 | nontext_est=332 | gen=87 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: Round, royal, residence, Berkshire, seen | keywords=24 | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** description sentences=9 | keywords=24
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=773 | text_est=442 | nontext_est=331 | gen=321 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=70% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1497 | text_est=442 | nontext_est=1055 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: which, indicates | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=474 | text_est=442 | nontext_est=32 | gen=149 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|end|&gt; appeared in generated text. | Special control token &lt;|endoftext|&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=66%
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=39
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1319 | text_est=442 | nontext_est=877 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=93% | missing sections: title, description, keywords | missing terms: royal, residence, Berkshire, which, indicates
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6618 | text_est=442 | nontext_est=6176 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=90% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4603 | text_est=442 | nontext_est=4161 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=87% | missing sections: title, description, keywords | missing terms: view, royal, residence, Berkshire, seen
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3399 | text_est=442 | nontext_est=2957 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=93% | missing sections: title, description | missing terms: view, royal, residence, Berkshire, seen
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description | keywords=55
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6618 | text_est=442 | nontext_est=6176 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=0.99% | nontext prompt burden=97% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=16
- **Utility:** user=caveat | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16720 | text_est=442 | nontext_est=16278 | gen=166 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=84% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2786 | text_est=442 | nontext_est=2344 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=76% | missing sections: title, description, keywords | missing terms: photograph
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1818 | text_est=442 | nontext_est=1376 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=81% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2286 | text_est=442 | nontext_est=1844 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** At long prompt length (16722 tokens), output became repetitive. | hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16722 | text_est=442 | nontext_est=16280 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect long-context cache behavior under heavy image-token burden.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** token_cap | user=recommended
- **Why:** At long prompt length (16731 tokens), output may stop following prompt/image context. | hit token cap (500) | nontext prompt burden=97% | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** title words=3 | keywords=109 | keyword duplication=0.43
- **Utility:** user=recommended | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16731 | text_est=442 | nontext_est=16289 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect long-context cache behavior under heavy image-token burden.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=71% | missing sections: title, description, keywords | missing terms: view, Tower, Windsor, Castle, royal
- **Trusted hints:** ignores trusted hints | missing terms: view, Tower, Windsor, Castle, royal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1531 | text_est=442 | nontext_est=1089 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: view, Round, Tower, royal, residence
- **Trusted hints:** degrades trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | degrades trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1690 | text_est=442 | nontext_est=1248 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=93% | missing sections: title, description | missing terms: view, royal, residence, Berkshire, seen
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description | keywords=44 | keyword duplication=0.52
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6618 | text_est=442 | nontext_est=6176 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=87% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, royal
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, royal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3308 | text_est=442 | nontext_est=2866 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: royal, Berkshire, photograph
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=442 | nontext_est=16304 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | context echo=100%
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=442 | nontext_est=16304 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: description, keywords | missing terms: royal, residence, Berkshire, seen, across
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=47
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=442 | nontext_est=16304 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1690 | text_est=442 | nontext_est=1248 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|endoftext|&gt; appeared in generated text. | Output appears truncated to about 2 tokens. | nontext prompt burden=97% | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=16731 | text_est=442 | nontext_est=16289 | gen=2 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: royal, residence, Berkshire, flagpole, which
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=442 | nontext_est=16304 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-4-31b-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: view, seen, across, flying, flagpole | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=767 | text_est=442 | nontext_est=325 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: description, keywords | missing terms: royal, Berkshire, seen, across, Thames
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=21
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=442 | nontext_est=16304 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: view, Berkshire, seen, across, River
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=442 | nontext_est=16304 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | missing sections: title, description, keywords | missing terms: residence, Berkshire, Union, indicates, reigning | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=475 | text_est=442 | nontext_est=33 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.
