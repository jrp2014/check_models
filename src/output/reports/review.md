# Automated Review Digest

_Generated on 2026-05-03 21:53:13 BST_

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
| `mlx-vlm`                        | `stop_token`                            |                      1 | `microsoft/Phi-3.5-vision-instruct`                     | [`mlx-vlm_stop-token_001`](../issues/issue_004_mlx-vlm_stop-token_001.md)                                                                                                     | Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete. |
| `model-config / mlx-vlm`         | `prompt_template`                       |                      3 | `mlx-community/llava-v1.6-mistral-7b-8bit`              | [`model-config-mlx-vlm_prompt-template_001`](../issues/issue_005_model-config-mlx-vlm_prompt-template_001.md)                                                                 | Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.          |
| `model-config / mlx-vlm`         | `stop_token`                            |                      1 | `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [`model-config-mlx-vlm_stop-token_001`](../issues/issue_006_model-config-mlx-vlm_stop-token_001.md)                                                                           | Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete. |
| `mlx-vlm / mlx`                  | `long_context`                          |                      1 | `mlx-community/X-Reasoner-7B-8bit`                      | [`mlx-vlm-mlx_long-context_001`](../issues/issue_007_mlx-vlm-mlx_long-context_001.md)                                                                                         | A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.               |
| `mlx-vlm / mlx`                  | `long_context`                          |                      2 | `mlx-community/Qwen3.5-27B-4bit`                        | [`mlx-vlm-mlx_long-context_002`](../issues/issue_008_mlx-vlm-mlx_long-context_002.md)                                                                                         | A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.               |

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (86/100) | Desc 100 | Keywords 86 | Δ+12 | 113.9 tps
- `mlx-community/gemma-3-27b-it-qat-4bit`: 🏆 A (85/100) | Desc 95 | Keywords 86 | Δ+11 | 31.4 tps

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | Δ-68 | 28.2 tps | harness, missing sections
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (6/100) | Desc 36 | Keywords 0 | Δ-68 | 196.5 tps | context ignored, hallucination, harness
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) | Desc 45 | Keywords 0 | Δ-63 | 59.1 tps | context ignored, harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (18/100) | Desc 51 | Keywords 32 | Δ-56 | 5.1 tps | context ignored, harness, missing sections
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (32/100) | Desc 81 | Keywords 75 | Δ-41 | 18.4 tps | context ignored, harness, missing sections

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                   | Verdict   | Hint Handling           | Key Evidence                                                                                    |
|-----------------------------------------|-----------|-------------------------|-------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit` | `clean`   | preserves trusted hints | missing terms: 10 Best (structured), Bird, Gull, Marina, Mudflats                               |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `clean`   | preserves trusted hints | missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring                                |
| `mlx-community/pixtral-12b-8bit`        | `clean`   | preserves trusted hints | nontext prompt burden=86% \| missing terms: 10 Best (structured), Bird, Mooring, view, historic |
| `mlx-community/pixtral-12b-bf16`        | `clean`   | preserves trusted hints | nontext prompt burden=86% \| missing terms: 10 Best (structured), Bird, Mooring, view, historic |

### `caveat`

| Model                                               | Verdict          | Hint Handling                                        | Key Evidence                                                                                                                                      |
|-----------------------------------------------------|------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=3.88% \| nontext prompt burden=85% \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring \| nonvisual metadata reused |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=4.00% \| nontext prompt burden=85% \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring \| nonvisual metadata reused |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                                          | Key Evidence                                                                                                                                                                                                                                                      |
|---------------------------------------------------------|---------------------|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `runtime_failure`   | not evaluated                                                                                                          | model error \| mlx model load model                                                                                                                                                                                                                               |
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure`   | not evaluated                                                                                                          | processor error \| model config processor load processor                                                                                                                                                                                                          |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | missing sections: keywords \| missing terms: 10 Best (structured), Bird, Blue sky, Gull, Marina \| context echo=95% \| nonvisual metadata reused                                                                                                                  |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring \| keywords=19 \| context echo=44% \| nonvisual metadata reused                                                                                                                                  |
| `qnguyen3/nanoLLaVA`                                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | missing sections: keywords \| missing terms: 10 Best (structured), Bird, Blue sky, Gull, Marina \| context echo=66% \| nonvisual metadata reused                                                                                                                  |
| `mlx-community/FastVLM-0.5B-bf16`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | missing sections: keywords \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring \| nonvisual metadata reused                                                                                                                                       |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `harness`           | ignores trusted hints \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                              | Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | keywords=20 \| context echo=60% \| nonvisual metadata reused                                                                                                                                                                                                      |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `model_shortcoming` | ignores trusted hints \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                              | nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                                                                                                                   |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `model_shortcoming` | ignores trusted hints \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull \| nonvisual metadata reused | nontext prompt burden=73% \| missing sections: title \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull \| keywords=26                                                                                                                           |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `model_shortcoming` | ignores trusted hints \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull \| nonvisual metadata reused | nontext prompt burden=73% \| missing sections: title \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull \| keywords=26                                                                                                                           |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                   | hit token cap (500) \| missing terms: 10 Best (structured), Mudflats \| keyword duplication=82% \| nonvisual metadata reused                                                                                                                                      |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `harness`           | ignores trusted hints \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                              | Output was a short generic filler response (about 8 tokens). \| nontext prompt burden=83% \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                                                                                                     |
| `mlx-community/InternVL3-8B-bf16`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | nontext prompt burden=80% \| missing terms: 10 Best (structured), Bird, Mooring, Mudflats, Pier \| nonvisual metadata reused                                                                                                                                      |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | nontext prompt burden=66% \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring \| keywords=20 \| nonvisual metadata reused                                                                                                                         |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | `clean`             | degrades trusted hints                                                                                                 | nontext prompt burden=85% \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring                                                                                                                                                                     |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull \| nonvisual metadata reused                                                                                            |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `harness`           | ignores trusted hints \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                              | Output is very short relative to prompt size (0.9%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | nontext prompt burden=83% \| missing terms: 10 Best (structured), Bird, Blue sky, Gull, Marina \| context echo=51% \| nonvisual metadata reused                                                                                                                   |
| `mlx-community/gemma-4-31b-it-4bit`                     | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | missing terms: 10 Best (structured), Bird, Marina, River Deben, view \| nonvisual metadata reused                                                                                                                                                                 |
| `mlx-community/InternVL3-14B-8bit`                      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | nontext prompt burden=80% \| missing terms: 10 Best (structured), Bird, view, historic, Suffolk \| nonvisual metadata reused                                                                                                                                      |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | missing sections: title, description, keywords \| missing terms: 10 Best (structured), Bird, Gull, Mudflats, Museum \| nonvisual metadata reused                                                                                                                  |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | missing terms: 10 Best (structured), Bird, Blue sky, Gull, Marina \| nonvisual metadata reused                                                                                                                                                                    |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | preserves trusted hints                                                                                                | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 56 occurrences). \| nontext prompt burden=82% \| missing sections: description, keywords \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring                           |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                   | hit token cap (500) \| nontext prompt burden=69% \| missing sections: title, description \| missing terms: 10 Best (structured), Quay, view, historic, sunny                                                                                                      |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                                   | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=66%                                                                  |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                                                                                            |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | degrades trusted hints                                                                                                 | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                                                                                            |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                   | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring                                                                                                                   |
| `mlx-community/gemma-4-31b-bf16`                        | `model_shortcoming` | ignores trusted hints \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                              | missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull \| formatting=Unknown tags: &lt;image&gt;                                                                                                      |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | missing sections: title, description, keywords \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring \| nonvisual metadata reused                                                                                                                   |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | preserves trusted hints                                                                                                | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring                                                                                            |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                   | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), view, sunny, day, including                                                                                            |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                                   | nontext prompt burden=73% \| keywords=19 \| nonvisual metadata reused                                                                                                                                                                                             |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `model_shortcoming` | preserves trusted hints                                                                                                | missing sections: title, description, keywords \| missing terms: 10 Best (structured), Bird, Blue sky, Gull, Mooring                                                                                                                                              |
| `mlx-community/X-Reasoner-7B-8bit`                      | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                   | At long prompt length (16807 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing terms: 10 Best (structured), Bird, Gull, Mooring, Mudflats                                                                         |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: 10 Best (structured), Bird, Blue sky, Gull, Marina                             | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Bird, Blue sky, Gull, Marina                                                                                           |
| `mlx-community/GLM-4.6V-nvfp4`                          | `cutoff_degraded`   | preserves trusted hints                                                                                                | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: 10 Best (structured), Bird, Mooring, Museum, River Deben                                                                                               |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | preserves trusted hints                                                                                                | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Marina, Mooring, Pier, Quay                                                                                            |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | preserves trusted hints                                                                                                | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring                                                                                            |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Gull, Marina, Pier, Quay                                                                                               |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | preserves trusted hints                                                                                                | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Marina, Mooring, Mudflats, Pier                                                                                        |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness`           | ignores trusted hints \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                              | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| Output appears truncated to about 6 tokens. \| nontext prompt burden=97% \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                                           |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring                                                                                            |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Marina, Mooring, Pier, Quay                                                                                            |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                                   | hit token cap (500) \| nontext prompt burden=69% \| missing sections: title, description \| missing terms: 10 Best (structured), view, historic, quayside, Suffolk                                                                                                |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints                                                                                                | hit token cap (500) \| nontext prompt burden=97% \| missing terms: 10 Best (structured), Gull, Marina, Pier, Quay \| keywords=21                                                                                                                                  |

## Maintainer Queue

Owner-grouped escalations only; clean recommended models stay in user buckets.

### `mlx`

| Model                                     | Verdict           | Evidence                                                                                                                                                                                  | Next Action                                                             |
|-------------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `runtime_failure` | model error \| mlx model load model                                                                                                                                                       | Inspect KV/cache behavior, memory pressure, and long-context execution. |
| `mlx-community/X-Reasoner-7B-8bit`        | `cutoff_degraded` | At long prompt length (16807 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing terms: 10 Best (structured), Bird, Gull, Mooring, Mudflats | Inspect long-context cache behavior under heavy image-token burden.     |

### `mlx-vlm`

| Model                                                   | Verdict   | Evidence                                                                                                                                                                                                                                | Next Action                                                                         |
|---------------------------------------------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 56 occurrences). \| nontext prompt burden=82% \| missing sections: description, keywords \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring | Inspect decode cleanup; tokenizer markers are leaking into user-facing text.        |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=66%                                        | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness` | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| Output appears truncated to about 6 tokens. \| nontext prompt burden=97% \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                 | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model                                               | Verdict             | Evidence                                                                                                                                                                   | Next Action                                                                                                        |
|-----------------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `model_shortcoming` | missing sections: keywords \| missing terms: 10 Best (structured), Bird, Blue sky, Gull, Marina \| context echo=95% \| nonvisual metadata reused                           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/LFM2-VL-1.6B-8bit`                   | `model_shortcoming` | missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring \| keywords=19 \| context echo=44% \| nonvisual metadata reused                                           | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `qnguyen3/nanoLLaVA`                                | `model_shortcoming` | missing sections: keywords \| missing terms: 10 Best (structured), Bird, Blue sky, Gull, Marina \| context echo=66% \| nonvisual metadata reused                           | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`          | `model_shortcoming` | keywords=20 \| context echo=60% \| nonvisual metadata reused                                                                                                               | Treat as a model-quality limitation for this prompt and image.                                                     |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `context_budget`    | output/prompt=3.88% \| nontext prompt burden=85% \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring \| nonvisual metadata reused                          | Treat this as a prompt-budget issue first; nontext prompt burden is 85% and the output stays weak under that load. |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `model_shortcoming` | nontext prompt burden=73% \| missing sections: title \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull \| keywords=26                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/SmolVLM-Instruct-bf16`               | `model_shortcoming` | nontext prompt burden=73% \| missing sections: title \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull \| keywords=26                                    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                 | `cutoff_degraded`   | hit token cap (500) \| missing terms: 10 Best (structured), Mudflats \| keyword duplication=82% \| nonvisual metadata reused                                               | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `mlx-community/gemma-3n-E2B-4bit`                   | `cutoff_degraded`   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull \| nonvisual metadata reused     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `context_budget`    | output/prompt=4.00% \| nontext prompt burden=85% \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring \| nonvisual metadata reused                          | Treat this as a prompt-budget issue first; nontext prompt burden is 85% and the output stays weak under that load. |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `model_shortcoming` | nontext prompt burden=83% \| missing terms: 10 Best (structured), Bird, Blue sky, Gull, Marina \| context echo=51% \| nonvisual metadata reused                            | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=69% \| missing sections: title, description \| missing terms: 10 Best (structured), Quay, view, historic, sunny               | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/paligemma2-3b-pt-896-4bit`           | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`     | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring                            | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/gemma-4-31b-bf16`                    | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull \| formatting=Unknown tags: &lt;image&gt;               | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-Flash-6bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`  | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), view, sunny, day, including     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Molmo-7B-D-0924-bf16`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Bird, Blue sky, Gull, Marina    | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/GLM-4.6V-nvfp4`                      | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: 10 Best (structured), Bird, Mooring, Museum, River Deben        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Marina, Mooring, Pier, Quay     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Gull, Marina, Pier, Quay        | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Marina, Mooring, Mudflats, Pier | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.6-27B-mxfp8`                   | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Bird, Gull, Marina, Mooring     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-4bit`                    | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Marina, Mooring, Pier, Quay     | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=69% \| missing sections: title, description \| missing terms: 10 Best (structured), view, historic, quayside, Suffolk         | Treat as a model limitation for this prompt; the requested output contract is not being met.                       |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | `cutoff_degraded`   | hit token cap (500) \| nontext prompt burden=97% \| missing terms: 10 Best (structured), Gull, Marina, Pier, Quay \| keywords=21                                           | Treat as a model limitation for this prompt; trusted hint coverage is still weak.                                  |

### `model-config`

| Model                                            | Verdict           | Evidence                                                                                                                                                                                                                                                          | Next Action                                                                                          |
|--------------------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `mlx-community/MolmoPoint-8B-fp16`               | `runtime_failure` | processor error \| model config processor load processor                                                                                                                                                                                                          | Inspect model repo config, chat template, and EOS settings.                                          |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`  | `harness`         | Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull | Check chat-template and EOS defaults first; the output shape is not matching the requested contract. |
| `mlx-community/llava-v1.6-mistral-7b-8bit`       | `harness`         | Output was a short generic filler response (about 8 tokens). \| nontext prompt burden=83% \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull                                                                                                     | Inspect model repo config, chat template, and EOS settings.                                          |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness`         | Output is very short relative to prompt size (0.9%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull | Check chat-template and EOS defaults first; the output shape is not matching the requested contract. |

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
- _Key signals:_ missing sections: keywords; missing terms: 10 Best
  (structured), Bird, Blue sky, Gull, Marina; context echo=95%; nonvisual
  metadata reused
- _Tokens:_ prompt 579 tok; estimated text 483 tok; estimated non-text 96 tok;
  generated 85 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: 10 Best (structured), Bird, Gull, Marina,
  Mooring; keywords=19; context echo=44%; nonvisual metadata reused
- _Tokens:_ prompt 834 tok; estimated text 483 tok; estimated non-text 351
  tok; generated 146 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: 10 Best
  (structured), Bird, Blue sky, Gull, Marina; context echo=66%; nonvisual
  metadata reused
- _Tokens:_ prompt 579 tok; estimated text 483 tok; estimated non-text 96 tok;
  generated 92 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: 10 Best
  (structured), Bird, Gull, Marina, Mooring; nonvisual metadata reused
- _Tokens:_ prompt 583 tok; estimated text 483 tok; estimated non-text 100
  tok; generated 175 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Check chat-template and EOS defaults first; the output shape is
  not matching the requested contract.
- _Key signals:_ Output is very short relative to prompt size (0.7%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=70%; missing sections: title, description, keywords; missing terms:
  10 Best (structured), Barge, Bird, Blue sky, Gull
- _Tokens:_ prompt 1596 tok; estimated text 483 tok; estimated non-text 1113
  tok; generated 11 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: 10 Best (structured), Bird, Gull, Marina,
  Mudflats
- _Tokens:_ prompt 844 tok; estimated text 483 tok; estimated non-text 361
  tok; generated 91 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; context echo=60%; nonvisual metadata reused
- _Tokens:_ prompt 691 tok; estimated text 483 tok; estimated non-text 208
  tok; generated 150 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 85% and the output stays weak under that load.
- _Key signals:_ output/prompt=3.88%; nontext prompt burden=85%; missing
  terms: 10 Best (structured), Bird, Gull, Marina, Mooring; nonvisual metadata
  reused
- _Tokens:_ prompt 3171 tok; estimated text 483 tok; estimated non-text 2688
  tok; generated 123 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title,
  description, keywords; missing terms: 10 Best (structured), Barge, Bird,
  Blue sky, Gull
- _Tokens:_ prompt 1596 tok; estimated text 483 tok; estimated non-text 1113
  tok; generated 20 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=73%; missing sections: title; missing
  terms: 10 Best (structured), Barge, Bird, Blue sky, Gull; keywords=26
- _Tokens:_ prompt 1790 tok; estimated text 483 tok; estimated non-text 1307
  tok; generated 254 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=73%; missing sections: title; missing
  terms: 10 Best (structured), Barge, Bird, Blue sky, Gull; keywords=26
- _Tokens:_ prompt 1790 tok; estimated text 483 tok; estimated non-text 1307
  tok; generated 254 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: 10 Best (structured),
  Mudflats; keyword duplication=82%; nonvisual metadata reused
- _Tokens:_ prompt 644 tok; estimated text 483 tok; estimated non-text 161
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output was a short generic filler response (about 8 tokens).;
  nontext prompt burden=83%; missing terms: 10 Best (structured), Barge, Bird,
  Blue sky, Gull
- _Tokens:_ prompt 2790 tok; estimated text 483 tok; estimated non-text 2307
  tok; generated 8 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=80%; missing terms: 10 Best
  (structured), Bird, Mooring, Mudflats, Pier; nonvisual metadata reused
- _Tokens:_ prompt 2362 tok; estimated text 483 tok; estimated non-text 1879
  tok; generated 105 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=66%; missing terms: 10 Best
  (structured), Bird, Gull, Marina, Mooring; keywords=20; nonvisual metadata
  reused
- _Tokens:_ prompt 1407 tok; estimated text 483 tok; estimated non-text 924
  tok; generated 195 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ avoid for now; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=85%; missing terms: 10 Best
  (structured), Bird, Gull, Marina, Mooring
- _Tokens:_ prompt 3172 tok; estimated text 483 tok; estimated non-text 2689
  tok; generated 112 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull;
  nonvisual metadata reused
- _Tokens:_ prompt 830 tok; estimated text 483 tok; estimated non-text 347
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: 10 Best (structured), Bird, Gull, Marina,
  Mooring
- _Tokens:_ prompt 839 tok; estimated text 483 tok; estimated non-text 356
  tok; generated 98 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Check chat-template and EOS defaults first; the output shape is
  not matching the requested contract.
- _Key signals:_ Output is very short relative to prompt size (0.9%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=70%; missing sections: title, description, keywords; missing terms:
  10 Best (structured), Barge, Bird, Blue sky, Gull
- _Tokens:_ prompt 1596 tok; estimated text 483 tok; estimated non-text 1113
  tok; generated 14 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 85% and the output stays weak under that load.
- _Key signals:_ output/prompt=4.00%; nontext prompt burden=85%; missing
  terms: 10 Best (structured), Bird, Gull, Marina, Mooring; nonvisual metadata
  reused
- _Tokens:_ prompt 3172 tok; estimated text 483 tok; estimated non-text 2689
  tok; generated 127 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=86%; missing terms: 10 Best
  (structured), Bird, Mooring, view, historic
- _Tokens:_ prompt 3362 tok; estimated text 483 tok; estimated non-text 2879
  tok; generated 114 tok; requested max 500 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=83%; missing terms: 10 Best
  (structured), Bird, Blue sky, Gull, Marina; context echo=51%; nonvisual
  metadata reused
- _Tokens:_ prompt 2859 tok; estimated text 483 tok; estimated non-text 2376
  tok; generated 129 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: 10 Best (structured), Bird, Marina, River
  Deben, view; nonvisual metadata reused
- _Tokens:_ prompt 844 tok; estimated text 483 tok; estimated non-text 361
  tok; generated 95 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=80%; missing terms: 10 Best
  (structured), Bird, view, historic, Suffolk; nonvisual metadata reused
- _Tokens:_ prompt 2362 tok; estimated text 483 tok; estimated non-text 1879
  tok; generated 119 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: 10 Best (structured), Bird, Gull, Mudflats, Museum; nonvisual
  metadata reused
- _Tokens:_ prompt 838 tok; estimated text 483 tok; estimated non-text 355
  tok; generated 336 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=86%; missing terms: 10 Best
  (structured), Bird, Mooring, view, historic
- _Tokens:_ prompt 3362 tok; estimated text 483 tok; estimated non-text 2879
  tok; generated 112 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: 10 Best (structured), Bird, Blue sky, Gull,
  Marina; nonvisual metadata reused
- _Tokens:_ prompt 839 tok; estimated text 483 tok; estimated non-text 356
  tok; generated 106 tok; requested max 500 tok; stop reason completed


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 56 occurrences).; nontext prompt burden=82%; missing sections:
  description, keywords; missing terms: 10 Best (structured), Bird, Gull,
  Marina, Mooring
- _Tokens:_ prompt 2674 tok; estimated text 483 tok; estimated non-text 2191
  tok; generated 97 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=69%; missing
  sections: title, description; missing terms: 10 Best (structured), Quay,
  view, historic, sunny
- _Tokens:_ prompt 1570 tok; estimated text 483 tok; estimated non-text 1087
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; Special control token &lt;|endoftext|&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=66%
- _Tokens:_ prompt 1407 tok; estimated text 483 tok; estimated non-text 924
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: 10 Best (structured),
  Barge, Bird, Blue sky, Gull
- _Tokens:_ prompt 4668 tok; estimated text 483 tok; estimated non-text 4185
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: 10 Best (structured),
  Barge, Bird, Blue sky, Gull
- _Tokens:_ prompt 6693 tok; estimated text 483 tok; estimated non-text 6210
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title; missing terms: 10 Best (structured), Bird, Gull, Marina,
  Mooring
- _Tokens:_ prompt 3453 tok; estimated text 483 tok; estimated non-text 2970
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: 10 Best (structured), Barge, Bird, Blue sky, Gull; formatting=Unknown
  tags: &lt;image&gt;
- _Tokens:_ prompt 832 tok; estimated text 483 tok; estimated non-text 349
  tok; generated 86 tok; requested max 500 tok; stop reason completed


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: 10 Best (structured), Bird, Gull, Marina, Mooring; nonvisual metadata
  reused
- _Tokens:_ prompt 548 tok; estimated text 483 tok; estimated non-text 65 tok;
  generated 235 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: 10 Best (structured),
  Bird, Gull, Marina, Mooring
- _Tokens:_ prompt 6693 tok; estimated text 483 tok; estimated non-text 6210
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description, keywords; missing terms: 10 Best (structured),
  view, sunny, day, including
- _Tokens:_ prompt 1896 tok; estimated text 483 tok; estimated non-text 1413
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%; keywords=19; nonvisual metadata
  reused
- _Tokens:_ prompt 1758 tok; estimated text 483 tok; estimated non-text 1275
  tok; generated 235 tok; requested max 500 tok; stop reason completed


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: 10 Best (structured), Bird, Blue sky, Gull, Mooring
- _Tokens:_ prompt 549 tok; estimated text 483 tok; estimated non-text 66 tok;
  generated 114 tok; requested max 500 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16807 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing terms:
  10 Best (structured), Bird, Gull, Mooring, Mudflats
- _Tokens:_ prompt 16807 tok; estimated text 483 tok; estimated non-text 16324
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: 10 Best (structured),
  Bird, Blue sky, Gull, Marina
- _Tokens:_ prompt 1758 tok; estimated text 483 tok; estimated non-text 1275
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description; missing terms: 10 Best (structured), Bird,
  Mooring, Museum, River Deben
- _Tokens:_ prompt 6693 tok; estimated text 483 tok; estimated non-text 6210
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: 10 Best (structured),
  Marina, Mooring, Pier, Quay
- _Tokens:_ prompt 16821 tok; estimated text 483 tok; estimated non-text 16338
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: 10 Best (structured),
  Bird, Gull, Marina, Mooring
- _Tokens:_ prompt 16821 tok; estimated text 483 tok; estimated non-text 16338
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: 10 Best (structured),
  Gull, Marina, Pier, Quay
- _Tokens:_ prompt 16821 tok; estimated text 483 tok; estimated non-text 16338
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: 10 Best (structured),
  Marina, Mooring, Mudflats, Pier
- _Tokens:_ prompt 16821 tok; estimated text 483 tok; estimated non-text 16338
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; Output appears truncated to about 6 tokens.; nontext prompt
  burden=97%; missing terms: 10 Best (structured), Barge, Bird, Blue sky, Gull
- _Tokens:_ prompt 16807 tok; estimated text 483 tok; estimated non-text 16324
  tok; generated 6 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: 10 Best (structured),
  Bird, Gull, Marina, Mooring
- _Tokens:_ prompt 16821 tok; estimated text 483 tok; estimated non-text 16338
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: 10 Best (structured),
  Marina, Mooring, Pier, Quay
- _Tokens:_ prompt 16821 tok; estimated text 483 tok; estimated non-text 16338
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=69%; missing
  sections: title, description; missing terms: 10 Best (structured), view,
  historic, quayside, Suffolk
- _Tokens:_ prompt 1570 tok; estimated text 483 tok; estimated non-text 1087
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: 10 Best (structured), Gull, Marina, Pier, Quay; keywords=21
- _Tokens:_ prompt 16821 tok; estimated text 483 tok; estimated non-text 16338
  tok; generated 500 tok; requested max 500 tok; stop reason completed

