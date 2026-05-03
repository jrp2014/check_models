# Automated Review Digest

_Generated on 2026-05-03 02:16:08 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

## Issue Queue

Root-cause issue drafts are queued in [issues/index.md](../issues/index.md).

| Owner                            | Issue Subtype                           |   Affected Model Count | Representative Model                                    | Issue Draft                                                                                                                                                                   | Acceptance Signal                                                                                                                              |
|----------------------------------|-----------------------------------------|------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx`                            | `MLX_MODEL_LOAD_MODEL`                  |                      1 | `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | [`mlx_mlx-model-load-model_001`](../issues/issue_001_mlx_mlx-model-load-model_001.md)                                                                                         | Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer. |
| `model configuration/repository` | `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR` |                      1 | `mlx-community/MolmoPoint-8B-fp16`                      | [`model-configuration-repository_model-config-processor-load-processor_001`](../issues/issue_002_model-configuration-repository_model-config-processor-load-processor_001.md) | Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer. |
| `mlx-vlm`                        | `encoding`                              |                      1 | `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [`mlx-vlm_encoding_001`](../issues/issue_003_mlx-vlm_encoding_001.md)                                                                                                         | Affected reruns contain no leaked BPE, byte-level, or tokenizer marker text.                                                                   |
| `mlx-vlm`                        | `stop_token`                            |                      2 | `microsoft/Phi-3.5-vision-instruct`                     | [`mlx-vlm_stop-token_001`](../issues/issue_004_mlx-vlm_stop-token_001.md)                                                                                                     | Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete. |
| `model-config / mlx-vlm`         | `prompt_template`                       |                      2 | `mlx-community/llava-v1.6-mistral-7b-8bit`              | [`model-config-mlx-vlm_prompt-template_001`](../issues/issue_005_model-config-mlx-vlm_prompt-template_001.md)                                                                 | Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.          |
| `model-config / mlx-vlm`         | `stop_token`                            |                      1 | `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [`model-config-mlx-vlm_stop-token_001`](../issues/issue_006_model-config-mlx-vlm_stop-token_001.md)                                                                           | Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete. |
| `mlx-vlm / mlx`                  | `long_context`                          |                      1 | `mlx-community/X-Reasoner-7B-8bit`                      | [`mlx-vlm-mlx_long-context_001`](../issues/issue_007_mlx-vlm-mlx_long-context_001.md)                                                                                         | A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.               |
| `mlx-vlm / mlx`                  | `long_context`                          |                      1 | `mlx-community/Qwen3.5-27B-4bit`                        | [`mlx-vlm-mlx_long-context_002`](../issues/issue_008_mlx-vlm-mlx_long-context_002.md)                                                                                         | A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.               |

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (86/100) | Desc 89 | Keywords 86 | Δ+11 | 47.5 tps
- `mlx-community/gemma-3-27b-it-qat-4bit`: 🏆 A (85/100) | Desc 100 | Keywords 84 | Δ+10 | 28.2 tps
- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (83/100) | Desc 100 | Keywords 85 | Δ+8 | 110.5 tps
- `mlx-community/gemma-4-31b-it-4bit`: ✅ B (80/100) | Desc 100 | Keywords 87 | Δ+4 | 26.5 tps
- `mlx-community/pixtral-12b-bf16`: ✅ B (79/100) | Desc 93 | Keywords 85 | Δ+4 | 19.1 tps

### Watchlist

- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-75 | harness
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) | Desc 21 | Keywords 0 | Δ-70 | 288.9 tps | context ignored, harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 40 | Keywords 0 | Δ-69 | 31.4 tps | harness, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (18/100) | Desc 51 | Keywords 32 | Δ-57 | 5.5 tps | context ignored, harness, missing sections
- `microsoft/Phi-3.5-vision-instruct`: 🟡 C (58/100) | Desc 89 | Keywords 58 | Δ-17 | 55.4 tps | hallucination, harness, metadata borrowing

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict   | Hint Handling           | Key Evidence                                                                                          |
|-----------------------------------------------------|-----------|-------------------------|-------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`   | preserves trusted hints | missing terms: Bench, East Anglia, English countryside, Mill, Moored                                  |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`   | preserves trusted hints | missing terms: East Anglia, English countryside, Moored, Objects, River Deben                         |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `clean`   | preserves trusted hints | missing terms: East Anglia, English countryside, Moored, Objects, River Deben                         |
| `mlx-community/pixtral-12b-bf16`                    | `clean`   | preserves trusted hints | nontext prompt burden=86% \| missing terms: East Anglia, English countryside, Moored, Objects, Quay   |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`   | preserves trusted hints | nontext prompt burden=85% \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside |

### `caveat`

| Model                                               | Verdict          | Hint Handling                                        | Key Evidence                                                                                                                                                         |
|-----------------------------------------------------|------------------|------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=4.31% \| nontext prompt burden=85% \| missing terms: Bench, Blue sky, Clouds, English countryside, Mudflats \| nonvisual metadata reused               |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `clean`          | preserves trusted hints                              | nontext prompt burden=83% \| missing terms: Bench, East Anglia, English countryside, Mill, Moored \| keywords=9 \| formatting=Unknown tags: &lt;end_of_utterance&gt; |
| `mlx-community/pixtral-12b-8bit`                    | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=2.85% \| nontext prompt burden=86% \| missing terms: East Anglia, English countryside, Moored, Objects, Quay \| nonvisual metadata reused              |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=3.74% \| nontext prompt burden=85% \| missing terms: Bench, East Anglia, English countryside, Moored, Mudflats \| nonvisual metadata reused            |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                                                  | Key Evidence                                                                                                                                                                                                                                                              |
|---------------------------------------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `runtime_failure`   | not evaluated                                                                                                                  | model error \| mlx model load model                                                                                                                                                                                                                                       |
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure`   | not evaluated                                                                                                                  | processor error \| model config processor load processor                                                                                                                                                                                                                  |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | missing sections: keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside \| context echo=66% \| nonvisual metadata reused                                                                                                                   |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | missing terms: Suffolk, seen, foreground \| keywords=20 \| nonvisual metadata reused                                                                                                                                                                                      |
| `qnguyen3/nanoLLaVA`                                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | missing sections: keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside \| nonvisual metadata reused                                                                                                                                       |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | missing terms: Bench, Blue sky, Clouds, Moored, Objects \| keywords=22 \| nonvisual metadata reused                                                                                                                                                                       |
| `mlx-community/FastVLM-0.5B-bf16`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | missing sections: keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Mudflats \| nonvisual metadata reused                                                                                                                                       |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | keywords=20 \| context echo=55% \| nonvisual metadata reused                                                                                                                                                                                                              |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `model_shortcoming` | ignores trusted hints \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                              | nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                                                                                                                   |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `harness`           | ignores trusted hints                                                                                                          | Model returned zero output tokens.                                                                                                                                                                                                                                        |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | nontext prompt burden=65% \| missing terms: Bench, Blue sky, Moored, Objects \| keywords=22 \| context echo=45%                                                                                                                                                           |
| `mlx-community/InternVL3-8B-bf16`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | nontext prompt burden=80% \| missing terms: Bench, Clouds, English countryside, Moored, Objects \| nonvisual metadata reused                                                                                                                                              |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `harness`           | ignores trusted hints \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                              | Output is very short relative to prompt size (0.9%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside \| nonvisual metadata reused                                                                                            |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `cutoff_degraded`   | degrades trusted hints                                                                                                         | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                                                                                            |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `cutoff_degraded`   | degrades trusted hints                                                                                                         | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                                                                                            |
| `mlx-community/InternVL3-14B-8bit`                      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | nontext prompt burden=80% \| missing terms: Bench, Clouds, East Anglia, English countryside, Objects \| nonvisual metadata reused                                                                                                                                         |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | missing sections: title, description, keywords \| missing terms: Bench, English countryside, Moored, Objects, Woodbridge \| nonvisual metadata reused                                                                                                                     |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | preserves trusted hints                                                                                                        | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 78 occurrences). \| nontext prompt burden=82% \| missing sections: description, keywords \| missing terms: Bench, Blue sky, East Anglia, English countryside, Moored                           |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | missing terms: Bench, English countryside, Moored, Objects, River Deben \| nonvisual metadata reused                                                                                                                                                                      |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                           | hit token cap (500) \| nontext prompt burden=69% \| missing sections: title \| missing terms: Clouds, English countryside, Moored, Mudflats, Objects                                                                                                                      |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                                           | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=65%                                                                          |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Mudflats \| nonvisual metadata reused                                                                                                                   |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                           | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Mudflats                                                                                                         |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                           | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title, description \| missing terms: English countryside, Moored, Objects, Quay, Rope                                                                                                               |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `harness`           | preserves trusted hints                                                                                                        | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, keywords                                                                                                                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                                                                                            |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | preserves trusted hints                                                                                                        | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                                                                                             |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | nontext prompt burden=73% \| missing terms: English countryside, seen, foreground \| nonvisual metadata reused                                                                                                                                                            |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | nontext prompt burden=73% \| missing terms: English countryside, seen, foreground \| nonvisual metadata reused                                                                                                                                                            |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                           | missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Mudflats, Objects \| nonvisual metadata reused                                                                                                                  |
| `mlx-community/X-Reasoner-7B-8bit`                      | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                           | At long prompt length (16794 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing terms: Bench, English countryside, Moored, Mudflats, Objects                                                                               |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `cutoff_degraded`   | degrades trusted hints                                                                                                         | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                                                                                            |
| `mlx-community/GLM-4.6V-nvfp4`                          | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                           | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Objects, River Deben                                                                                        |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                           | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                                                                                             |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                           | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                                                                                             |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | preserves trusted hints                                                                                                        | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                                                                                             |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness`           | ignores trusted hints \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                              | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| Output appears truncated to about 2 tokens. \| nontext prompt burden=97% \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                                           |
| `mlx-community/gemma-4-31b-bf16`                        | `cutoff_degraded`   | preserves trusted hints                                                                                                        | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, East Anglia, English countryside, Moored \| degeneration=incomplete_sentence: ends with 'in'                                                                     |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | preserves trusted hints                                                                                                        | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                                                                                             |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                           | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| nonvisual metadata reused                                                                                                                                           |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints                                                                                                        | hit token cap (500) \| nontext prompt burden=97% \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                                                                                                                                               |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                           | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| nonvisual metadata reused                                                                                                                                           |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                           | hit token cap (500) \| nontext prompt burden=69% \| missing sections: title, description \| missing terms: Woodbridge, Suffolk, seen, foreground, long                                                                                                                    |

## Maintainer Queue

Owner-grouped escalations only; clean recommended models stay in user buckets.

### `mlx`

| Model                                     | Verdict           | Evidence                                                                                                                                                                                    | Next Action                                                             |
|-------------------------------------------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `runtime_failure` | model error \| mlx model load model                                                                                                                                                         | Inspect KV/cache behavior, memory pressure, and long-context execution. |
| `mlx-community/X-Reasoner-7B-8bit`        | `cutoff_degraded` | At long prompt length (16794 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing terms: Bench, English countryside, Moored, Mudflats, Objects | Inspect long-context cache behavior under heavy image-token burden.     |

### `mlx-vlm`

| Model                                                   | Verdict   | Evidence                                                                                                                                                                                                                                        | Next Action                                                                         |
|---------------------------------------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 78 occurrences). \| nontext prompt burden=82% \| missing sections: description, keywords \| missing terms: Bench, Blue sky, East Anglia, English countryside, Moored | Inspect decode cleanup; tokenizer markers are leaking into user-facing text.        |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=65%                                                | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, keywords                                                                                       | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness` | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| Output appears truncated to about 2 tokens. \| nontext prompt burden=97% \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                 | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model                                               | Verdict             | Evidence                                                                                                                                                                                              | Next Action                                                                                                        |
|-----------------------------------------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `model_shortcoming` | missing sections: keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside \| context echo=66% \| nonvisual metadata reused                                               | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`          | `model_shortcoming` | keywords=20 \| context echo=55% \| nonvisual metadata reused                                                                                                                                          | Treat as a model-quality limitation for this prompt and image.                                                     |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `context_budget`    | output/prompt=4.31% \| nontext prompt burden=85% \| missing terms: Bench, Blue sky, Clouds, English countryside, Mudflats \| nonvisual metadata reused                                                | Treat this as a prompt-budget issue first; nontext prompt burden is 85% and the output stays weak under that load. |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        | `model_shortcoming` | nontext prompt burden=65% \| missing terms: Bench, Blue sky, Moored, Objects \| keywords=22 \| context echo=45%                                                                                       | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `clean`             | nontext prompt burden=83% \| missing terms: Bench, East Anglia, English countryside, Mill, Moored \| keywords=9 \| formatting=Unknown tags: &lt;end_of_utterance&gt;                                  | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/gemma-3n-E2B-4bit`                   | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside \| nonvisual metadata reused                        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/SmolVLM-Instruct-bf16`               | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/pixtral-12b-8bit`                    | `context_budget`    | output/prompt=2.85% \| nontext prompt burden=86% \| missing terms: East Anglia, English countryside, Moored, Objects, Quay \| nonvisual metadata reused                                               | Treat this as a prompt-budget issue first; nontext prompt burden is 86% and the output stays weak under that load. |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `context_budget`    | output/prompt=3.74% \| nontext prompt burden=85% \| missing terms: Bench, East Anglia, English countryside, Moored, Mudflats \| nonvisual metadata reused                                             | Treat this as a prompt-budget issue first; nontext prompt burden is 85% and the output stays weak under that load. |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=69% \| missing sections: title \| missing terms: Clouds, English countryside, Moored, Mudflats, Objects                                                  | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`     | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Mudflats                                     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`  | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title, description \| missing terms: English countryside, Moored, Objects, Quay, Rope                                           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/paligemma2-3b-pt-896-4bit`           | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-Flash-6bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`     | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside                        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-nvfp4`                      | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Objects, River Deben                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-4-31b-bf16`                    | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, East Anglia, English countryside, Moored \| degeneration=incomplete_sentence: ends with 'in' | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-4bit`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| nonvisual metadata reused                                                                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing terms: Bench, East Anglia, English countryside, Moored, Objects                                                                           | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/Qwen3.6-27B-mxfp8`                   | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| nonvisual metadata reused                                                                       | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=69% \| missing sections: title, description \| missing terms: Woodbridge, Suffolk, seen, foreground, long                                                | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |

### `model-config`

| Model                                            | Verdict           | Evidence                                                                                                                                                                                                                                                                  | Next Action                                                                                          |
|--------------------------------------------------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `mlx-community/MolmoPoint-8B-fp16`               | `runtime_failure` | processor error \| model config processor load processor                                                                                                                                                                                                                  | Inspect model repo config, chat template, and EOS settings.                                          |
| `mlx-community/llava-v1.6-mistral-7b-8bit`       | `harness`         | Model returned zero output tokens.                                                                                                                                                                                                                                        | Inspect model repo config, chat template, and EOS settings.                                          |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness`         | Output is very short relative to prompt size (0.9%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Bench, Blue sky, Clouds, East Anglia, English countryside | Check chat-template and EOS defaults first; the output shape is not matching the requested contract. |

## Model Verdicts

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Model
  Error`; diagnostic code `MLX_MODEL_LOAD_MODEL`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ model error; mlx model load model
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `model-config`; reported package `model-config`;
  failure stage `Processor Error`; diagnostic code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ processor error; model config processor load processor
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside; context echo=66%; nonvisual
  metadata reused
- _Tokens:_ prompt 566 tok; estimated text 481 tok; estimated non-text 85 tok;
  generated 86 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Suffolk, seen, foreground; keywords=20;
  nonvisual metadata reused
- _Tokens:_ prompt 826 tok; estimated text 481 tok; estimated non-text 345
  tok; generated 115 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside; nonvisual metadata reused
- _Tokens:_ prompt 566 tok; estimated text 481 tok; estimated non-text 85 tok;
  generated 74 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bench, Blue sky, Clouds, Moored, Objects;
  keywords=22; nonvisual metadata reused
- _Tokens:_ prompt 636 tok; estimated text 481 tok; estimated non-text 155
  tok; generated 198 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bench, East Anglia, English countryside, Mill,
  Moored
- _Tokens:_ prompt 837 tok; estimated text 481 tok; estimated non-text 356
  tok; generated 83 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bench, East
  Anglia, English countryside, Moored, Mudflats; nonvisual metadata reused
- _Tokens:_ prompt 570 tok; estimated text 481 tok; estimated non-text 89 tok;
  generated 221 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; context echo=55%; nonvisual metadata reused
- _Tokens:_ prompt 679 tok; estimated text 481 tok; estimated non-text 198
  tok; generated 134 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title,
  description, keywords; missing terms: Bench, Blue sky, Clouds, East Anglia,
  English countryside
- _Tokens:_ prompt 1587 tok; estimated text 481 tok; estimated non-text 1106
  tok; generated 20 tok; requested max 500 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Model returned zero output tokens.
- _Tokens:_ prompt 0 tok; estimated text 481 tok; estimated non-text 0 tok;
  generated 0 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 85% and the output stays weak under that load.
- _Key signals:_ output/prompt=4.31%; nontext prompt burden=85%; missing
  terms: Bench, Blue sky, Clouds, English countryside, Mudflats; nonvisual
  metadata reused
- _Tokens:_ prompt 3179 tok; estimated text 481 tok; estimated non-text 2698
  tok; generated 137 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=65%; missing terms: Bench, Blue sky,
  Moored, Objects; keywords=22; context echo=45%
- _Tokens:_ prompt 1394 tok; estimated text 481 tok; estimated non-text 913
  tok; generated 169 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=80%; missing terms: Bench, Clouds,
  English countryside, Moored, Objects; nonvisual metadata reused
- _Tokens:_ prompt 2349 tok; estimated text 481 tok; estimated non-text 1868
  tok; generated 103 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Check chat-template and EOS defaults first; the output shape is
  not matching the requested contract.
- _Key signals:_ Output is very short relative to prompt size (0.9%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=70%; missing sections: title, description, keywords; missing terms:
  Bench, Blue sky, Clouds, East Anglia, English countryside
- _Tokens:_ prompt 1587 tok; estimated text 481 tok; estimated non-text 1106
  tok; generated 14 tok; requested max 500 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=83%; missing terms: Bench, East Anglia,
  English countryside, Mill, Moored; keywords=9; formatting=Unknown tags:
  &lt;end_of_utterance&gt;
- _Tokens:_ prompt 2848 tok; estimated text 481 tok; estimated non-text 2367
  tok; generated 91 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bench, Blue sky, Clouds, East Anglia, English
  countryside; nonvisual metadata reused
- _Tokens:_ prompt 823 tok; estimated text 481 tok; estimated non-text 342
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside
- _Tokens:_ prompt 1779 tok; estimated text 481 tok; estimated non-text 1298
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside
- _Tokens:_ prompt 1779 tok; estimated text 481 tok; estimated non-text 1298
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=80%; missing terms: Bench, Clouds, East
  Anglia, English countryside, Objects; nonvisual metadata reused
- _Tokens:_ prompt 2349 tok; estimated text 481 tok; estimated non-text 1868
  tok; generated 108 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: East Anglia, English countryside, Moored,
  Objects, River Deben
- _Tokens:_ prompt 837 tok; estimated text 481 tok; estimated non-text 356
  tok; generated 100 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ output/prompt=2.85%; nontext prompt burden=86%; missing
  terms: East Anglia, English countryside, Moored, Objects, Quay; nonvisual
  metadata reused
- _Tokens:_ prompt 3373 tok; estimated text 481 tok; estimated non-text 2892
  tok; generated 96 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: East Anglia, English countryside, Moored,
  Objects, River Deben
- _Tokens:_ prompt 832 tok; estimated text 481 tok; estimated non-text 351
  tok; generated 108 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=86%; missing terms: East Anglia,
  English countryside, Moored, Objects, Quay
- _Tokens:_ prompt 3373 tok; estimated text 481 tok; estimated non-text 2892
  tok; generated 90 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 85% and the output stays weak under that load.
- _Key signals:_ output/prompt=3.74%; nontext prompt burden=85%; missing
  terms: Bench, East Anglia, English countryside, Moored, Mudflats; nonvisual
  metadata reused
- _Tokens:_ prompt 3180 tok; estimated text 481 tok; estimated non-text 2699
  tok; generated 119 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=85%; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside
- _Tokens:_ prompt 3180 tok; estimated text 481 tok; estimated non-text 2699
  tok; generated 119 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bench, English countryside, Moored, Objects, Woodbridge; nonvisual
  metadata reused
- _Tokens:_ prompt 831 tok; estimated text 481 tok; estimated non-text 350
  tok; generated 326 tok; requested max 500 tok; stop reason completed


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 78 occurrences).; nontext prompt burden=82%; missing sections:
  description, keywords; missing terms: Bench, Blue sky, East Anglia, English
  countryside, Moored
- _Tokens:_ prompt 2682 tok; estimated text 481 tok; estimated non-text 2201
  tok; generated 124 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bench, English countryside, Moored, Objects,
  River Deben; nonvisual metadata reused
- _Tokens:_ prompt 832 tok; estimated text 481 tok; estimated non-text 351
  tok; generated 116 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=69%; missing
  sections: title; missing terms: Clouds, English countryside, Moored,
  Mudflats, Objects
- _Tokens:_ prompt 1559 tok; estimated text 481 tok; estimated non-text 1078
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; Special control token &lt;|endoftext|&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=65%
- _Tokens:_ prompt 1394 tok; estimated text 481 tok; estimated non-text 913
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bench, East Anglia, English countryside, Moored, Mudflats; nonvisual
  metadata reused
- _Tokens:_ prompt 536 tok; estimated text 481 tok; estimated non-text 55 tok;
  generated 100 tok; requested max 500 tok; stop reason completed


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, keywords; missing terms: Bench, East Anglia, English
  countryside, Moored, Mudflats
- _Tokens:_ prompt 3464 tok; estimated text 481 tok; estimated non-text 2983
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description; missing terms: English countryside, Moored,
  Objects, Quay, Rope
- _Tokens:_ prompt 1887 tok; estimated text 481 tok; estimated non-text 1406
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=93%; missing sections:
  title, keywords
- _Tokens:_ prompt 6681 tok; estimated text 481 tok; estimated non-text 6200
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside
- _Tokens:_ prompt 4659 tok; estimated text 481 tok; estimated non-text 4178
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Moored, Objects
- _Tokens:_ prompt 6681 tok; estimated text 481 tok; estimated non-text 6200
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=73%; missing terms: English
  countryside, seen, foreground; nonvisual metadata reused
- _Tokens:_ prompt 1753 tok; estimated text 481 tok; estimated non-text 1272
  tok; generated 190 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=73%; missing terms: English
  countryside, seen, foreground; nonvisual metadata reused
- _Tokens:_ prompt 1753 tok; estimated text 481 tok; estimated non-text 1272
  tok; generated 190 tok; requested max 500 tok; stop reason completed


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bench, East Anglia, English countryside, Mudflats, Objects; nonvisual
  metadata reused
- _Tokens:_ prompt 537 tok; estimated text 481 tok; estimated non-text 56 tok;
  generated 116 tok; requested max 500 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16794 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing terms:
  Bench, English countryside, Moored, Mudflats, Objects
- _Tokens:_ prompt 16794 tok; estimated text 481 tok; estimated non-text 16313
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=70%; missing
  sections: title, description, keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside
- _Tokens:_ prompt 1587 tok; estimated text 481 tok; estimated non-text 1106
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Objects, River Deben
- _Tokens:_ prompt 6681 tok; estimated text 481 tok; estimated non-text 6200
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Moored, Objects
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Moored, Objects
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Moored, Objects
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; Output appears truncated to about 2 tokens.; nontext prompt
  burden=97%; missing terms: Bench, Blue sky, Clouds, East Anglia, English
  countryside
- _Tokens:_ prompt 16794 tok; estimated text 481 tok; estimated non-text 16313
  tok; generated 2 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bench, Blue sky, East Anglia, English countryside,
  Moored; degeneration=incomplete_sentence: ends with 'in'
- _Tokens:_ prompt 825 tok; estimated text 481 tok; estimated non-text 344
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Moored, Objects
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; nonvisual metadata reused
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: Bench, East Anglia, English countryside, Moored, Objects
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; nonvisual metadata reused
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=69%; missing
  sections: title, description; missing terms: Woodbridge, Suffolk, seen,
  foreground, long
- _Tokens:_ prompt 1559 tok; estimated text 481 tok; estimated non-text 1078
  tok; generated 500 tok; requested max 500 tok; stop reason completed

