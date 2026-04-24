# Automated Review Digest

_Generated on 2026-04-24 22:50:35 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (96/100) | Desc 100 | Keywords 77 | Δ+50 | 113.7 tps
- `mlx-community/gemma-4-31b-it-4bit`: 🏆 A (94/100) | Desc 100 | Keywords 83 | Δ+49 | 27.2 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (92/100) | Desc 100 | Keywords 91 | Δ+47 | 66.0 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (90/100) | Desc 93 | Keywords 91 | Δ+45 | 62.9 tps
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🏆 A (90/100) | Desc 93 | Keywords 85 | Δ+45 | 17.4 tps

### Watchlist

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) | Desc 21 | Keywords 0 | Δ-40 | 290.1 tps | context ignored, harness
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (6/100) | Desc 45 | Keywords 0 | Δ-39 | 70.0 tps | context ignored, harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | Δ-39 | 31.6 tps | harness, missing sections
- `microsoft/Phi-3.5-vision-instruct`: 🟡 C (62/100) | Desc 60 | Keywords 55 | Δ+17 | 55.1 tps | hallucination, harness, metadata borrowing
- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (15/100) | Desc 60 | Keywords 0 | Δ-30 | 128.9 tps | metadata borrowing, missing sections

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict   | Hint Handling          | Key Evidence                                                                          |
|-----------------------------------------------------|-----------|------------------------|---------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`   | improves trusted hints | missing terms: view, Round, Windsor, royal, residence                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`   | improves trusted hints | nontext prompt burden=86% \| missing terms: royal, residence, Berkshire, seen, across |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`   | improves trusted hints | nontext prompt burden=86% \| missing terms: seen, which, indicates, reigning, monarch |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`   | improves trusted hints | missing terms: view, Round, Windsor, royal, residence                                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `clean`   | improves trusted hints | missing terms: royal, residence, Berkshire, seen, flagpole                            |

### `caveat`

| Model                                             | Verdict          | Hint Handling                                       | Key Evidence                                                                                                                              |
|---------------------------------------------------|------------------|-----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `context_budget` | improves trusted hints \| nonvisual metadata reused | output/prompt=3.56% \| nontext prompt burden=86% \| missing terms: seen, across, flying, indicates, reigning \| nonvisual metadata reused |
| `mlx-community/gemma-3-27b-it-qat-4bit`           | `clean`          | improves trusted hints                              | missing terms: royal, residence, Berkshire, seen, which \| keywords=19                                                                    |
| `mlx-community/pixtral-12b-8bit`                  | `context_budget` | improves trusted hints \| nonvisual metadata reused | output/prompt=3.02% \| nontext prompt burden=87% \| missing terms: view, royal, seen, across, flying \| nonvisual metadata reused         |
| `mlx-community/InternVL3-14B-8bit`                | `clean`          | improves trusted hints                              | nontext prompt burden=81% \| missing terms: royal, residence, Berkshire, seen, across                                                     |
| `mlx-community/pixtral-12b-bf16`                  | `context_budget` | improves trusted hints \| nonvisual metadata reused | output/prompt=2.90% \| nontext prompt burden=87% \| missing terms: view, royal, seen, across, flying \| nonvisual metadata reused         |
| `mlx-community/X-Reasoner-7B-8bit`                | `context_budget` | improves trusted hints \| nonvisual metadata reused | output/prompt=0.81% \| nontext prompt burden=97% \| missing terms: seen, which, indicates, reigning, monarch \| keywords=22               |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                            | Key Evidence                                                                                                                                                                                                                   |
|---------------------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ggml-org/gemma-3-1b-it-GGUF`                           | `runtime_failure`   | not evaluated                                                                                            | model error \| model config model load model                                                                                                                                                                                   |
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure`   | not evaluated                                                                                            | processor error \| model config processor load processor                                                                                                                                                                       |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | degrades trusted hints \| nonvisual metadata reused                                                      | missing sections: keywords \| missing terms: view, Round, residence, Berkshire, seen \| nonvisual metadata reused                                                                                                              |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | missing terms: view, seen, which, indicates, reigning \| nonvisual metadata reused                                                                                                                                             |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Union, Flag, flying, flagpole \| nonvisual metadata reused                                                                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Union, Flag, flying, flagpole \| nonvisual metadata reused                                                                 |
| `qnguyen3/nanoLLaVA`                                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | missing sections: keywords \| context echo=54% \| nonvisual metadata reused                                                                                                                                                    |
| `mlx-community/FastVLM-0.5B-bf16`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | missing sections: keywords \| missing terms: which, indicates \| nonvisual metadata reused                                                                                                                                     |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | missing terms: which, indicates \| keywords=6 \| nonvisual metadata reused                                                                                                                                                     |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | nontext prompt burden=66% \| missing terms: which, indicates \| nonvisual metadata reused                                                                                                                                      |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `harness`           | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | Output was a short generic filler response (about 8 tokens). \| nontext prompt burden=84% \| missing terms: view, Round, Tower, Windsor, Castle                                                                                |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `model_shortcoming` | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                                              |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                     | hit token cap (500) \| keyword duplication=91% \| nonvisual metadata reused \| repetitive token=phrase: "flagpole, flag, flagpole, flag..."                                                                                    |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                      | nontext prompt burden=70% \| missing sections: title \| missing terms: flagpole, which, indicates, reigning, monarch \| keywords=22                                                                                            |
| `mlx-community/InternVL3-8B-bf16`                       | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                      | nontext prompt burden=81% \| missing terms: view, royal, residence, Berkshire, seen \| nonvisual metadata reused                                                                                                               |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `model_shortcoming` | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                                              |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: which, indicates, reigning, photograph                                                                    |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title \| missing terms: royal, indicates, reigning, photograph                                                                                           |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle \| nonvisual metadata reused                                                                       |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | preserves trusted hints                                                                                  | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 58 occurrences). \| nontext prompt burden=83% \| missing sections: description, keywords \| missing terms: view, royal, residence, Berkshire, which |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                      | missing terms: Round, Tower, royal, residence, Berkshire \| keywords=34 \| nonvisual metadata reused                                                                                                                           |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                      | nontext prompt burden=84% \| missing terms: flying, which, indicates, photograph \| keywords=38 \| nonvisual metadata reused                                                                                                   |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `model_shortcoming` | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                                              |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | missing sections: title, description, keywords \| missing terms: which, indicates \| nonvisual metadata reused                                                                                                                 |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                     | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=66%                               |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=76% \| missing sections: title, description \| missing terms: royal, residence, Berkshire, seen, indicates                                                                        |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                                                       |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: royal, residence, Berkshire, which, indicates                                                             |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: view, royal, residence, Berkshire, seen                                                                   |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: view, royal, residence, Berkshire, seen                                                                             |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                     | missing sections: title, description, keywords \| missing terms: view, which, indicates \| nonvisual metadata reused                                                                                                           |
| `mlx-community/GLM-4.6V-nvfp4`                          | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: view, royal, residence, Berkshire, seen                                                                             |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, seen, which, indicates, reigning                                                                    |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `cutoff_degraded`   | improves trusted hints                                                                                   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Windsor, royal, residence, Berkshire                                                                |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: royal, residence, Berkshire, flagpole, which                                                              |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | improves trusted hints                                                                                   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: view, royal, residence, Berkshire, Thames                                                                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| nonvisual metadata reused                                                                                                |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness`           | ignores trusted hints \| missing terms: view, Round, Tower, Windsor, Castle                              | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| Output appears truncated to about 2 tokens. \| nontext prompt burden=97% \| missing terms: view, Round, Tower, Windsor, Castle                      |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: view, Berkshire, seen, across, which                                                                      |
| `mlx-community/gemma-4-31b-bf16`                        | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, seen, across, flying, flagpole \| nonvisual metadata reused                                                                      |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | improves trusted hints                                                                                   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: royal, Berkshire, seen, across, flagpole                                                                  |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=97% \| missing terms: royal, residence, Berkshire, across, flagpole \| nonvisual metadata reused                                                                                  |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `mlx-vlm`

| Model                                                   | Verdict   | Evidence                                                                                                                                                                                                                       | Next Action                                                                         |
|---------------------------------------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 58 occurrences). \| nontext prompt burden=83% \| missing sections: description, keywords \| missing terms: view, royal, residence, Berkshire, which | Inspect decode cleanup; tokenizer markers are leaking into user-facing text.        |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=66%                               | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness` | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| Output appears truncated to about 2 tokens. \| nontext prompt burden=97% \| missing terms: view, Round, Tower, Windsor, Castle                      | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model                                               | Verdict             | Evidence                                                                                                                                                           | Next Action                                                                                                        |
|-----------------------------------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `model_shortcoming` | missing sections: keywords \| missing terms: view, Round, residence, Berkshire, seen \| nonvisual metadata reused                                                  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/LFM2-VL-1.6B-8bit`                   | `model_shortcoming` | missing terms: view, seen, which, indicates, reigning \| nonvisual metadata reused                                                                                 | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `model_shortcoming` | nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Union, Flag, flying, flagpole \| nonvisual metadata reused     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/SmolVLM-Instruct-bf16`               | `model_shortcoming` | nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Union, Flag, flying, flagpole \| nonvisual metadata reused     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `qnguyen3/nanoLLaVA`                                | `model_shortcoming` | missing sections: keywords \| context echo=54% \| nonvisual metadata reused                                                                                        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/FastVLM-0.5B-bf16`                   | `model_shortcoming` | missing sections: keywords \| missing terms: which, indicates \| nonvisual metadata reused                                                                         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`          | `model_shortcoming` | missing terms: which, indicates \| keywords=6 \| nonvisual metadata reused                                                                                         | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`             | missing terms: view, Round, Windsor, royal, residence                                                                                                              | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `context_budget`    | output/prompt=3.56% \| nontext prompt burden=86% \| missing terms: seen, across, flying, indicates, reigning \| nonvisual metadata reused                          | Treat this as a prompt-budget issue first; nontext prompt burden is 86% and the output stays weak under that load. |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        | `model_shortcoming` | nontext prompt burden=66% \| missing terms: which, indicates \| nonvisual metadata reused                                                                          | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | `model_shortcoming` | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                 | `cutoff_degraded`   | hit token cap (500) \| keyword duplication=91% \| nonvisual metadata reused \| repetitive token=phrase: "flagpole, flag, flagpole, flag..."                        | Treat as a model-quality limitation for this prompt and image.                                                     |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `model_shortcoming` | nontext prompt burden=70% \| missing sections: title \| missing terms: flagpole, which, indicates, reigning, monarch \| keywords=22                                | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/InternVL3-8B-bf16`                   | `model_shortcoming` | nontext prompt burden=81% \| missing terms: view, royal, residence, Berkshire, seen \| nonvisual metadata reused                                                   | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`             | nontext prompt burden=86% \| missing terms: royal, residence, Berkshire, seen, across                                                                              | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`             | nontext prompt burden=86% \| missing terms: seen, which, indicates, reigning, monarch                                                                              | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `clean`             | missing terms: royal, residence, Berkshire, seen, which \| keywords=19                                                                                             | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`             | missing terms: view, Round, Windsor, royal, residence                                                                                                              | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/pixtral-12b-8bit`                    | `context_budget`    | output/prompt=3.02% \| nontext prompt burden=87% \| missing terms: view, royal, seen, across, flying \| nonvisual metadata reused                                  | Treat this as a prompt-budget issue first; nontext prompt burden is 87% and the output stays weak under that load. |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`     | `model_shortcoming` | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: which, indicates, reigning, photograph        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`             | nontext prompt burden=81% \| missing terms: royal, residence, Berkshire, seen, across                                                                              | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`           | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title \| missing terms: royal, indicates, reigning, photograph                               | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-3n-E2B-4bit`                   | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle \| nonvisual metadata reused           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/pixtral-12b-bf16`                    | `context_budget`    | output/prompt=2.90% \| nontext prompt burden=87% \| missing terms: view, royal, seen, across, flying \| nonvisual metadata reused                                  | Treat this as a prompt-budget issue first; nontext prompt burden is 87% and the output stays weak under that load. |
| `mlx-community/gemma-3n-E4B-it-bf16`                | `model_shortcoming` | missing terms: Round, Tower, royal, residence, Berkshire \| keywords=34 \| nonvisual metadata reused                                                               | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `model_shortcoming` | nontext prompt burden=84% \| missing terms: flying, which, indicates, photograph \| keywords=38 \| nonvisual metadata reused                                       | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `clean`             | missing terms: royal, residence, Berkshire, seen, flagpole                                                                                                         | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | `model_shortcoming` | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle                                  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: which, indicates \| nonvisual metadata reused                                                     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`  | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=76% \| missing sections: title, description \| missing terms: royal, residence, Berkshire, seen, indicates            | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/paligemma2-3b-pt-896-4bit`           | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: view, Round, Tower, Windsor, Castle           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: royal, residence, Berkshire, which, indicates | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`     | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: view, royal, residence, Berkshire, seen       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-Flash-6bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: view, royal, residence, Berkshire, seen                 | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/X-Reasoner-7B-8bit`                  | `context_budget`    | output/prompt=0.81% \| nontext prompt burden=97% \| missing terms: seen, which, indicates, reigning, monarch \| keywords=22                                        | Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load. |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: view, which, indicates \| nonvisual metadata reused                                               | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-nvfp4`                      | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: view, royal, residence, Berkshire, seen                 | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Molmo-7B-D-0924-8bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, seen, which, indicates, reigning        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Molmo-7B-D-0924-bf16`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: view, Windsor, royal, residence, Berkshire    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: royal, residence, Berkshire, flagpole, which  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: view, royal, residence, Berkshire, Thames     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| nonvisual metadata reused                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: view, Berkshire, seen, across, which          | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-4-31b-bf16`                    | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: view, seen, across, flying, flagpole \| nonvisual metadata reused          | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-4bit`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: royal, Berkshire, seen, across, flagpole      | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing terms: royal, residence, Berkshire, across, flagpole \| nonvisual metadata reused                      | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |

### `model-config`

| Model                                      | Verdict           | Evidence                                                                                                                                        | Next Action                                                 |
|--------------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `ggml-org/gemma-3-1b-it-GGUF`              | `runtime_failure` | model error \| model config model load model                                                                                                    | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/MolmoPoint-8B-fp16`         | `runtime_failure` | processor error \| model config processor load processor                                                                                        | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `harness`         | Output was a short generic filler response (about 8 tokens). \| nontext prompt burden=84% \| missing terms: view, Round, Tower, Windsor, Castle | Inspect model repo config, chat template, and EOS settings. |

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
- **Why:** missing terms: view, seen, which, indicates, reigning | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=4 | description sentences=3
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=758 | text_est=442 | nontext_est=316 | gen=131 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: view, Union, Flag, flying, flagpole | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1716 | text_est=442 | nontext_est=1274 | gen=30 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: view, Union, Flag, flying, flagpole | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1716 | text_est=442 | nontext_est=1274 | gen=30 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `qnguyen3/nanoLLaVA`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: keywords | context echo=54% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords | title words=13
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=503 | text_est=442 | nontext_est=61 | gen=83 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: keywords | missing terms: which, indicates | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords | title words=16 | description sentences=3
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=507 | text_est=442 | nontext_est=65 | gen=158 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: which, indicates | keywords=6 | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=3 | keywords=6
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=616 | text_est=442 | nontext_est=174 | gen=92 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

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

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=66% | missing terms: which, indicates | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1317 | text_est=442 | nontext_est=875 | gen=110 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** harness | user=avoid
- **Why:** Output was a short generic filler response (about 8 tokens). | nontext prompt burden=84% | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=2722 | text_est=442 | nontext_est=2280 | gen=8 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=71% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1531 | text_est=442 | nontext_est=1089 | gen=24 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | keyword duplication=91% | nonvisual metadata reused | repetitive token=phrase: "flagpole, flag, flagpole, flag..."
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=14 | keywords=157 | keyword duplication=0.91
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=568 | text_est=442 | nontext_est=126 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=70% | missing sections: title | missing terms: flagpole, which, indicates, reigning, monarch | keywords=22
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title | keywords=22
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1497 | text_est=442 | nontext_est=1055 | gen=139 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=81% | missing terms: view, royal, residence, Berkshire, seen | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=4
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2286 | text_est=442 | nontext_est=1844 | gen=73 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=86% | missing terms: royal, residence, Berkshire, seen, across
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3115 | text_est=442 | nontext_est=2673 | gen=97 | max=500 | stop=completed
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

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** clean | user=caveat
- **Why:** missing terms: royal, residence, Berkshire, seen, which | keywords=19
- **Trusted hints:** improves trusted hints
- **Contract:** keywords=19
- **Utility:** user=caveat | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=774 | text_est=442 | nontext_est=332 | gen=90 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-4-31b-it-4bit`

- **Verdict:** clean | user=recommended
- **Why:** missing terms: view, Round, Windsor, royal, residence
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=779 | text_est=442 | nontext_est=337 | gen=84 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=3.02% | nontext prompt burden=87% | missing terms: view, royal, seen, across, flying | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=caveat | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3308 | text_est=442 | nontext_est=2866 | gen=100 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 87% and the output stays weak under that load.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=71% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1531 | text_est=442 | nontext_est=1089 | gen=77 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=70% | missing sections: title, description, keywords | missing terms: which, indicates, reigning, photograph
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1497 | text_est=442 | nontext_est=1055 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=81% | missing terms: royal, residence, Berkshire, seen, across
- **Trusted hints:** improves trusted hints
- **Contract:** title words=4
- **Utility:** user=caveat | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2286 | text_est=442 | nontext_est=1844 | gen=101 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=70% | missing sections: title | missing terms: royal, indicates, reigning, photograph
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1497 | text_est=442 | nontext_est=1055 | gen=500 | max=500 | stop=completed
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

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 58 occurrences). | nontext prompt burden=83% | missing sections: description, keywords | missing terms: view, royal, residence, Berkshire, which
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: description, keywords | title words=60
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2654 | text_est=442 | nontext_est=2212 | gen=84 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=2.90% | nontext prompt burden=87% | missing terms: view, royal, seen, across, flying | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=caveat | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3308 | text_est=442 | nontext_est=2866 | gen=96 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 87% and the output stays weak under that load.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: Round, Tower, royal, residence, Berkshire | keywords=34 | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** description sentences=6 | keywords=34
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=773 | text_est=442 | nontext_est=331 | gen=312 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=84% | missing terms: flying, which, indicates, photograph | keywords=38 | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** description sentences=4 | keywords=38
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2786 | text_est=442 | nontext_est=2344 | gen=184 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** clean | user=recommended
- **Why:** missing terms: royal, residence, Berkshire, seen, flagpole
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=774 | text_est=442 | nontext_est=332 | gen=93 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=71% | missing sections: title, description, keywords | missing terms: view, Round, Tower, Windsor, Castle
- **Trusted hints:** ignores trusted hints | missing terms: view, Round, Tower, Windsor, Castle
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1531 | text_est=442 | nontext_est=1089 | gen=31 | max=500 | stop=completed
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
- **Contract:** keywords=36
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1317 | text_est=442 | nontext_est=875 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=76% | missing sections: title, description | missing terms: royal, residence, Berkshire, seen, indicates
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description | keywords=46 | keyword duplication=0.50
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1818 | text_est=442 | nontext_est=1376 | gen=500 | max=500 | stop=completed
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

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=93% | missing sections: title, description, keywords | missing terms: royal, residence, Berkshire, which, indicates
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6618 | text_est=442 | nontext_est=6176 | gen=500 | max=500 | stop=completed
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

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** context_budget | user=caveat
- **Why:** output/prompt=0.81% | nontext prompt burden=97% | missing terms: seen, which, indicates, reigning, monarch | keywords=22
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** keywords=22
- **Utility:** user=caveat | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16731 | text_est=442 | nontext_est=16289 | gen=136 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: view, which, indicates | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=475 | text_est=442 | nontext_est=33 | gen=118 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=93% | missing sections: title, description | missing terms: view, royal, residence, Berkshire, seen
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description | keywords=36
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6618 | text_est=442 | nontext_est=6176 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: view, seen, which, indicates, reigning
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1690 | text_est=442 | nontext_est=1248 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: view, Windsor, royal, residence, Berkshire
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1690 | text_est=442 | nontext_est=1248 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: royal, residence, Berkshire, flagpole, which
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=442 | nontext_est=16304 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: view, royal, residence, Berkshire, Thames
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=442 | nontext_est=16304 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=442 | nontext_est=16304 | gen=500 | max=500 | stop=completed
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
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: view, Berkshire, seen, across, which
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
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing sections: title, description, keywords | missing terms: royal, Berkshire, seen, across, flagpole
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=442 | nontext_est=16304 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff_degraded | user=avoid
- **Why:** hit token cap (500) | nontext prompt burden=97% | missing terms: royal, residence, Berkshire, across, flagpole | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=442 | nontext_est=16304 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.
