<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

_Generated on 2026-06-21 01:13:25 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (94/100) | Desc 94 | Keywords 92 | Δ+26 | 62.5 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (94/100) | Desc 91 | Keywords 91 | Δ+25 | 66.0 tps
- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (93/100) | Desc 79 | Keywords 83 | Δ+24 | 97.7 tps
- `mlx-community/Qwen3.5-35B-A3B-bf16`: 🏆 A (93/100) | Desc 95 | Keywords 74 | Δ+24 | 54.1 tps
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🏆 A (89/100) | Desc 84 | Keywords 81 | Δ+21 | 174.0 tps

### Watchlist

- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-69 | 118.0 tps | harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | Δ-63 | 24.1 tps | harness, missing sections
- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (9/100) | Desc 57 | Keywords 0 | Δ-60 | 125.3 tps | harness, missing sections, trusted hint degraded
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (9/100) | Desc 57 | Keywords 0 | Δ-60 | 118.8 tps | harness, missing sections, trusted hint degraded
- `mlx-community/X-Reasoner-7B-8bit`: 🟡 C (58/100) | Desc 78 | Keywords 32 | Δ-11 | 41.3 tps | cutoff, generation loop, harness, long context, metadata borrowing, repetitive

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                   | Verdict     | Hint Handling                                        | Key Evidence                                                                                                                               |
|-----------------------------------------|-------------|------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `token_cap` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=72% \| missing terms: Entrance, High Street, Path, Rural, Tranquil \| keyword duplication=45% |
| `mlx-community/GLM-4.6V-Flash-mxfp4`    | `token_cap` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=94% \| missing terms: Colchester, Rural, exterior \| keyword duplication=85%                  |
| `mlx-community/Qwen3.5-35B-A3B-bf16`    | `clean`     | preserves trusted hints                              | nontext prompt burden=97% \| missing terms: Colchester, Entrance, Essex, High Street, Path                                                 |

### `caveat`

| Model                                     | Verdict          | Hint Handling                                        | Key Evidence                                                                                                                                    |
|-------------------------------------------|------------------|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/pixtral-12b-8bit`          | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=2.69% \| nontext prompt burden=87% \| missing terms: Colchester, Essex, High Street, St, Peters \| keywords=20                    |
| `mlx-community/pixtral-12b-bf16`          | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=2.60% \| nontext prompt burden=87% \| missing terms: Colchester, Essex, High Street, St, Peters \| keywords=19                    |
| `mlx-community/GLM-4.6V-Flash-6bit`       | `context_budget` | preserves trusted hints                              | output/prompt=1.81% \| nontext prompt burden=94% \| missing sections: title \| missing terms: Colchester, Essex, High Street, Rural, Tranquil   |
| `Qwen/Qwen3-VL-2B-Instruct`               | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=0.67% \| nontext prompt burden=97% \| missing terms: exterior \| keywords=20                                                      |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16` | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=0.67% \| nontext prompt burden=97% \| missing terms: exterior \| keywords=20                                                      |
| `mlx-community/GLM-4.6V-nvfp4`            | `context_budget` | degrades trusted hints                               | output/prompt=1.34% \| nontext prompt burden=94% \| missing sections: title \| missing terms: Colchester, Entrance, Essex, High Street, Path    |
| `mlx-community/Qwen3.5-35B-A3B-4bit`      | `clean`          | preserves trusted hints                              | nontext prompt burden=97% \| missing terms: Colchester, Entrance, Essex, High Street, Path \| keywords=19                                       |
| `mlx-community/Qwen3.5-35B-A3B-6bit`      | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=0.59% \| nontext prompt burden=97% \| missing terms: Colchester, Essex, High Street, Rural, Tranquil \| nonvisual metadata reused |
| `mlx-community/Qwen3.5-9B-MLX-4bit`       | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=0.70% \| nontext prompt burden=97% \| missing terms: High Street, Path, Tranquil, exterior, St \| nonvisual metadata reused       |
| `mlx-community/Qwen3.5-27B-mxfp8`         | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=0.59% \| nontext prompt burden=97% \| missing terms: Entrance, Essex, High Street, Rural, Tranquil \| nonvisual metadata reused   |
| `mlx-community/Qwen3.5-27B-4bit`          | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=0.75% \| nontext prompt burden=97% \| missing terms: Entrance, Essex, High Street, Path, Rural \| nonvisual metadata reused       |
| `mlx-community/Qwen3.6-27B-mxfp8`         | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=0.73% \| nontext prompt burden=97% \| missing terms: Colchester, Entrance, Essex, High Street, Path \| nonvisual metadata reused  |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                                    | Key Evidence                                                                                                                                                                                                                                               |
|---------------------------------------------------------|---------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `runtime_failure`   | not evaluated                                                                                                    | weight mismatch \| mlx model load weight mismatch                                                                                                                                                                                                          |
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure`   | not evaluated                                                                                                    | model error \| mlx vlm model load model                                                                                                                                                                                                                    |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing terms: Essex, High Street, Path, Shadows, Tranquil \| nonvisual metadata reused                                                                                                                                                                    |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | preserves trusted hints                                                                                          | missing sections: keywords \| missing terms: Clouds, Entrance, Fence, High Street, Path                                                                                                                                                                    |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `harness`           | degrades trusted hints                                                                                           | Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: Clouds, Colchester, Entrance, Essex, Fence |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | `harness`           | degrades trusted hints                                                                                           | Special control token &lt;/think&gt; appeared in generated text. \| nontext prompt burden=62% \| missing terms: Colchester, Entrance, Essex, High Street, Path \| reasoning leak                                                                           |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `harness`           | degrades trusted hints                                                                                           | Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: Clouds, Colchester, Entrance, Essex, Fence |
| `qnguyen3/nanoLLaVA`                                    | `model_shortcoming` | degrades trusted hints                                                                                           | missing sections: keywords \| missing terms: Colchester, Entrance, Essex, High Street, Path                                                                                                                                                                |
| `mlx-community/FastVLM-0.5B-bf16`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing sections: keywords \| missing terms: Entrance, High Street, Path, Rural, Shadows \| nonvisual metadata reused                                                                                                                                      |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | keywords=20 \| nonvisual metadata reused                                                                                                                                                                                                                   |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| missing terms: Entrance, High Street, Rural \| keyword duplication=92% \| nonvisual metadata reused                                                                                                                                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | `semantic_mismatch` | preserves trusted hints                                                                                          | missing terms: Colchester, Entrance, Essex, High Street, Path                                                                                                                                                                                              |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | `semantic_mismatch` | preserves trusted hints                                                                                          | nontext prompt burden=86% \| missing terms: Colchester, Entrance, Essex, Fence, High Street                                                                                                                                                                |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `model_shortcoming` | ignores trusted hints \| missing terms: Church, Clouds, Colchester, Entrance, Essex                              | nontext prompt burden=72% \| missing sections: title, description, keywords \| missing terms: Church, Clouds, Colchester, Entrance, Essex                                                                                                                  |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing terms: High Street, Path, Rural, Tranquil, Tree \| nonvisual metadata reused                                                                                                                                                                       |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing terms: High Street, Path, Rural, Tree, Trees \| nonvisual metadata reused                                                                                                                                                                          |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `harness`           | ignores trusted hints                                                                                            | Output appears truncated to about 2 tokens. \| nontext prompt burden=84%                                                                                                                                                                                   |
| `mlx-community/InternVL3-8B-bf16`                       | `semantic_mismatch` | preserves trusted hints                                                                                          | nontext prompt burden=81% \| missing terms: Colchester, Essex, High Street, Rural, Shadows                                                                                                                                                                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | nontext prompt burden=68% \| missing terms: Clouds, Path, Sky, exterior \| keywords=21 \| nonvisual metadata reused                                                                                                                                        |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | `semantic_mismatch` | preserves trusted hints                                                                                          | nontext prompt burden=86% \| missing terms: Colchester, Entrance, Essex, High Street, Path                                                                                                                                                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | `semantic_mismatch` | preserves trusted hints                                                                                          | nontext prompt burden=86% \| missing terms: Clouds, Colchester, Entrance, Essex, High Street                                                                                                                                                               |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Church, Clouds, Colchester, Entrance, Essex \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Church, Clouds, Colchester, Entrance, Essex \| nonvisual metadata reused                                                                                           |
| `mlx-community/gemma-4-31b-it-4bit`                     | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing terms: Entrance, High Street, Path, Rural, Tranquil \| nonvisual metadata reused                                                                                                                                                                   |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing terms: High Street, Rural, Tranquil, exterior, St \| nonvisual metadata reused                                                                                                                                                                     |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `model_shortcoming` | ignores trusted hints \| missing terms: Church, Clouds, Colchester, Entrance, Essex                              | nontext prompt burden=72% \| missing sections: title, description, keywords \| missing terms: Church, Clouds, Colchester, Entrance, Essex                                                                                                                  |
| `mlx-community/InternVL3-14B-8bit`                      | `semantic_mismatch` | preserves trusted hints                                                                                          | nontext prompt burden=81% \| missing terms: Colchester, Essex, High Street, Path, Rural                                                                                                                                                                    |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `semantic_mismatch` | preserves trusted hints                                                                                          | nontext prompt burden=85% \| missing terms: Entrance, Fence, High Street, Path, Rural \| keywords=9 \| formatting=Unknown tags: &lt;end_of_utterance&gt;                                                                                                   |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing sections: title, description, keywords \| missing terms: High Street, Shadows, exterior \| nonvisual metadata reused                                                                                                                               |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | preserves trusted hints                                                                                          | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 51 occurrences). \| nontext prompt burden=84% \| missing sections: description, keywords \| missing terms: Colchester, Entrance, Essex, High Street, exterior                   |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing terms: Entrance, High Street, Path, Rural, Tranquil \| nonvisual metadata reused                                                                                                                                                                   |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                             | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=68%                                                           |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=72% \| missing sections: title, description \| missing terms: Entrance, Path, exterior                                                                                                                        |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing sections: title, description, keywords \| missing terms: Entrance, High Street, Path, Rural, Tranquil \| nonvisual metadata reused                                                                                                                 |
| `mlx-community/gemma-4-31b-bf16`                        | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing sections: title, description, keywords \| missing terms: Entrance, High Street, Path, Rural, Shadows \| nonvisual metadata reused                                                                                                                  |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                             | missing sections: title, description, keywords \| missing terms: Entrance, High Street, Path, Shadows, Tranquil \| nonvisual metadata reused                                                                                                               |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=77% \| missing sections: title, description \| missing terms: Entrance, High Street                                                                                                                           |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | preserves trusted hints                                                                                          | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title \| missing terms: High Street, Rural, Tranquil                                                                                                                                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Church, Clouds, Colchester, Entrance, Essex                              | hit token cap (500) \| nontext prompt burden=91% \| missing sections: title, description, keywords \| missing terms: Church, Clouds, Colchester, Entrance, Essex                                                                                           |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `cutoff_degraded`   | degrades trusted hints                                                                                           | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: Clouds, Entrance, Fence, Path, Rural                                                                                                  |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `cutoff_degraded`   | ignores trusted hints \| missing terms: Church, Clouds, Colchester, Entrance, Essex                              | hit token cap (500) \| nontext prompt burden=72% \| missing sections: title, description, keywords \| missing terms: Church, Clouds, Colchester, Entrance, Essex                                                                                           |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description \| missing terms: Entrance, Tranquil, exterior                                                                                                                    |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `cutoff_degraded`   | degrades trusted hints                                                                                           | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: Church, Colchester, Essex, Fence, High Street                                                                                         |
| `mlx-community/X-Reasoner-7B-8bit`                      | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | At long prompt length (16736 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing terms: Clouds, High Street, Rural, Tree, Trees                                                                              |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | At long prompt length (16736 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing terms: Entrance, High Street, Path, Rural, Tranquil                                                                         |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                             | hit token cap (500) \| nontext prompt burden=72% \| missing sections: title \| missing terms: Clouds, Entrance, Essex, High Street, Path                                                                                                                   |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                                                                          | Evidence Snapshot                                                                                                                                                                                                                           | Affected Models                                            | Issue Draft                                                                                                                              | Evidence Bundle   | Fixed When                                                |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load                                                         | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                                                                           | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-weight-mismatch_001.md)   | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | mlx-vlm: Model load / model error: property 'eos_token_id' of 'ModelConfig' object has no setter | Model Error \| phase model_load \| AttributeError                                                                                                                                                                                           | 1: `mlx-community/MolmoPoint-8B-fp16`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_mlx-vlm-model-load-model_001.md)     | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers                                                         | 51 BPE space markers found in decoded text \| prompt=2,622 \| output/prompt=3.17% \| nontext burden=84% \| stop=completed                                                                                                                   | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm_encoding_001.md)                     | -                 | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text                                                   | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=1,328 \| output/prompt=37.65% \| nontext burden=68% \| stop=max_tokens \| hit token cap (500) \| 2 model cluster | 2: `microsoft/Phi-3.5-vision-instruct` (+1)                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_stop-token_001.md)                   | -                 | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                                                            | output/prompt=0.7% \| prompt=1,721 \| output/prompt=0.70% \| nontext burden=75% \| stop=completed \| 3 model cluster                                                                                                                        | 3: `HuggingFaceTB/SmolVLM-Instruct` (+2)                   | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_model-config-mlx-vlm_prompt-template_001.md) | -                 | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short                                            | prompt_tokens=16736, repetitive output \| prompt=16,736 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) \| 2 model cluster                                                                           | 2: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+1)          | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm-mlx_long-context_001.md)             | -                 | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

## Model Verdicts

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Weight
  Mismatch`; diagnostic code `MLX_MODEL_LOAD_WEIGHT_MISMATCH`
- _Next step:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.
- _Key signals:_ weight mismatch; mlx model load weight mismatch
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx-vlm`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect prompt-template, stop-token, and decode post-processing
  behavior.
- _Key signals:_ model error; mlx vlm model load model
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Essex, High Street, Path, Shadows, Tranquil;
  nonvisual metadata reused
- _Tokens:_ prompt 575 tok; estimated text 426 tok; estimated non-text 149
  tok; generated 117 tok; requested max 500 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Clouds, Entrance,
  Fence, High Street, Path
- _Tokens:_ prompt 508 tok; estimated text 426 tok; estimated non-text 82 tok;
  generated 37 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Check chat-template and EOS defaults first; the output shape is
  not matching the requested contract.
- _Key signals:_ Output is very short relative to prompt size (0.7%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=75%; missing sections: title, description, keywords; missing terms:
  Clouds, Colchester, Entrance, Essex, Fence
- _Tokens:_ prompt 1721 tok; estimated text 426 tok; estimated non-text 1295
  tok; generated 12 tok; requested max 500 tok; stop reason completed


### `mlx-community/MiniCPM-V-4.6-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; nontext prompt burden=62%; missing terms: Colchester, Entrance,
  Essex, High Street, Path; reasoning leak
- _Tokens:_ prompt 1123 tok; estimated text 426 tok; estimated non-text 697
  tok; generated 89 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Check chat-template and EOS defaults first; the output shape is
  not matching the requested contract.
- _Key signals:_ Output is very short relative to prompt size (0.7%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=75%; missing sections: title, description, keywords; missing terms:
  Clouds, Colchester, Entrance, Essex, Fence
- _Tokens:_ prompt 1721 tok; estimated text 426 tok; estimated non-text 1295
  tok; generated 12 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Colchester,
  Entrance, Essex, High Street, Path
- _Tokens:_ prompt 508 tok; estimated text 426 tok; estimated non-text 82 tok;
  generated 61 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Entrance, High
  Street, Path, Rural, Shadows; nonvisual metadata reused
- _Tokens:_ prompt 512 tok; estimated text 426 tok; estimated non-text 86 tok;
  generated 163 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; nonvisual metadata reused
- _Tokens:_ prompt 621 tok; estimated text 426 tok; estimated non-text 195
  tok; generated 95 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: Entrance, High Street,
  Rural; keyword duplication=92%; nonvisual metadata reused
- _Tokens:_ prompt 765 tok; estimated text 426 tok; estimated non-text 339
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Colchester, Entrance, Essex, High Street, Path
- _Tokens:_ prompt 781 tok; estimated text 426 tok; estimated non-text 355
  tok; generated 88 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=86%; missing terms: Colchester,
  Entrance, Essex, Fence, High Street
- _Tokens:_ prompt 3120 tok; estimated text 426 tok; estimated non-text 2694
  tok; generated 121 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=72%; missing sections: title,
  description, keywords; missing terms: Church, Clouds, Colchester, Entrance,
  Essex
- _Tokens:_ prompt 1532 tok; estimated text 426 tok; estimated non-text 1106
  tok; generated 29 tok; requested max 500 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: High Street, Path, Rural, Tranquil, Tree;
  nonvisual metadata reused
- _Tokens:_ prompt 781 tok; estimated text 426 tok; estimated non-text 355
  tok; generated 94 tok; requested max 500 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: High Street, Path, Rural, Tree, Trees;
  nonvisual metadata reused
- _Tokens:_ prompt 781 tok; estimated text 426 tok; estimated non-text 355
  tok; generated 95 tok; requested max 500 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 2 tokens.; nontext prompt
  burden=84%
- _Tokens:_ prompt 2730 tok; estimated text 426 tok; estimated non-text 2304
  tok; generated 2 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=81%; missing terms: Colchester, Essex,
  High Street, Rural, Shadows
- _Tokens:_ prompt 2291 tok; estimated text 426 tok; estimated non-text 1865
  tok; generated 69 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=68%; missing terms: Clouds, Path, Sky,
  exterior; keywords=21; nonvisual metadata reused
- _Tokens:_ prompt 1328 tok; estimated text 426 tok; estimated non-text 902
  tok; generated 177 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=86%; missing terms: Colchester,
  Entrance, Essex, High Street, Path
- _Tokens:_ prompt 3121 tok; estimated text 426 tok; estimated non-text 2695
  tok; generated 100 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=86%; missing terms: Clouds, Colchester,
  Entrance, Essex, High Street
- _Tokens:_ prompt 3121 tok; estimated text 426 tok; estimated non-text 2695
  tok; generated 110 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Church, Clouds, Colchester, Entrance, Essex;
  nonvisual metadata reused
- _Tokens:_ prompt 767 tok; estimated text 426 tok; estimated non-text 341
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 87% and the output stays weak under that load.
- _Key signals:_ output/prompt=2.69%; nontext prompt burden=87%; missing
  terms: Colchester, Essex, High Street, St, Peters; keywords=20
- _Tokens:_ prompt 3314 tok; estimated text 426 tok; estimated non-text 2888
  tok; generated 89 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Entrance, High Street, Path, Rural, Tranquil;
  nonvisual metadata reused
- _Tokens:_ prompt 781 tok; estimated text 426 tok; estimated non-text 355
  tok; generated 91 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: High Street, Rural, Tranquil, exterior, St;
  nonvisual metadata reused
- _Tokens:_ prompt 776 tok; estimated text 426 tok; estimated non-text 350
  tok; generated 95 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=72%; missing sections: title,
  description, keywords; missing terms: Church, Clouds, Colchester, Entrance,
  Essex
- _Tokens:_ prompt 1532 tok; estimated text 426 tok; estimated non-text 1106
  tok; generated 29 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=81%; missing terms: Colchester, Essex,
  High Street, Path, Rural
- _Tokens:_ prompt 2291 tok; estimated text 426 tok; estimated non-text 1865
  tok; generated 95 tok; requested max 500 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=85%; missing terms: Entrance, Fence,
  High Street, Path, Rural; keywords=9; formatting=Unknown tags:
  &lt;end_of_utterance&gt;
- _Tokens:_ prompt 2790 tok; estimated text 426 tok; estimated non-text 2364
  tok; generated 140 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 87% and the output stays weak under that load.
- _Key signals:_ output/prompt=2.60%; nontext prompt burden=87%; missing
  terms: Colchester, Essex, High Street, St, Peters; keywords=19
- _Tokens:_ prompt 3314 tok; estimated text 426 tok; estimated non-text 2888
  tok; generated 86 tok; requested max 500 tok; stop reason completed


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: High Street, Shadows, exterior; nonvisual metadata reused
- _Tokens:_ prompt 478 tok; estimated text 426 tok; estimated non-text 52 tok;
  generated 99 tok; requested max 500 tok; stop reason completed


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 51 occurrences).; nontext prompt burden=84%; missing sections:
  description, keywords; missing terms: Colchester, Entrance, Essex, High
  Street, exterior
- _Tokens:_ prompt 2622 tok; estimated text 426 tok; estimated non-text 2196
  tok; generated 83 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=72%; missing
  terms: Entrance, High Street, Path, Rural, Tranquil; keyword duplication=45%
- _Tokens:_ prompt 1502 tok; estimated text 426 tok; estimated non-text 1076
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Entrance, High Street, Path, Rural, Tranquil;
  nonvisual metadata reused
- _Tokens:_ prompt 776 tok; estimated text 426 tok; estimated non-text 350
  tok; generated 92 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; Special control token &lt;|endoftext|&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=68%
- _Tokens:_ prompt 1328 tok; estimated text 426 tok; estimated non-text 902
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=72%; missing
  sections: title, description; missing terms: Entrance, Path, exterior
- _Tokens:_ prompt 1502 tok; estimated text 426 tok; estimated non-text 1076
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Entrance, High Street, Path, Rural, Tranquil; nonvisual metadata
  reused
- _Tokens:_ prompt 775 tok; estimated text 426 tok; estimated non-text 349
  tok; generated 415 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Entrance, High Street, Path, Rural, Shadows; nonvisual metadata
  reused
- _Tokens:_ prompt 769 tok; estimated text 426 tok; estimated non-text 343
  tok; generated 50 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 94% and the output stays weak under that load.
- _Key signals:_ output/prompt=1.81%; nontext prompt burden=94%; missing
  sections: title; missing terms: Colchester, Essex, High Street, Rural,
  Tranquil
- _Tokens:_ prompt 6628 tok; estimated text 426 tok; estimated non-text 6202
  tok; generated 120 tok; requested max 500 tok; stop reason completed


### `Qwen/Qwen3-VL-2B-Instruct`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.67%; nontext prompt burden=97%; missing
  terms: exterior; keywords=20
- _Tokens:_ prompt 16725 tok; estimated text 426 tok; estimated non-text 16299
  tok; generated 112 tok; requested max 500 tok; stop reason completed


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Entrance, High Street, Path, Shadows, Tranquil; nonvisual metadata
  reused
- _Tokens:_ prompt 479 tok; estimated text 426 tok; estimated non-text 53 tok;
  generated 61 tok; requested max 500 tok; stop reason completed


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=77%; missing
  sections: title, description; missing terms: Entrance, High Street
- _Tokens:_ prompt 1826 tok; estimated text 426 tok; estimated non-text 1400
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=87%; missing
  sections: title; missing terms: High Street, Rural, Tranquil
- _Tokens:_ prompt 3405 tok; estimated text 426 tok; estimated non-text 2979
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: Church, Clouds,
  Colchester, Entrance, Essex
- _Tokens:_ prompt 4604 tok; estimated text 426 tok; estimated non-text 4178
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.67%; nontext prompt burden=97%; missing
  terms: exterior; keywords=20
- _Tokens:_ prompt 16725 tok; estimated text 426 tok; estimated non-text 16299
  tok; generated 112 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=94%; missing
  terms: Colchester, Rural, exterior; keyword duplication=85%
- _Tokens:_ prompt 6628 tok; estimated text 426 tok; estimated non-text 6202
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description, keywords; missing terms: Clouds, Entrance,
  Fence, Path, Rural
- _Tokens:_ prompt 1695 tok; estimated text 426 tok; estimated non-text 1269
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=72%; missing
  sections: title, description, keywords; missing terms: Church, Clouds,
  Colchester, Entrance, Essex
- _Tokens:_ prompt 1532 tok; estimated text 426 tok; estimated non-text 1106
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description; missing terms: Entrance, Tranquil, exterior
- _Tokens:_ prompt 16727 tok; estimated text 426 tok; estimated non-text 16301
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 94% and the output stays weak under that load.
- _Key signals:_ output/prompt=1.34%; nontext prompt burden=94%; missing
  sections: title; missing terms: Colchester, Entrance, Essex, High Street,
  Path
- _Tokens:_ prompt 6628 tok; estimated text 426 tok; estimated non-text 6202
  tok; generated 89 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description, keywords; missing terms: Church, Colchester,
  Essex, Fence, High Street
- _Tokens:_ prompt 1695 tok; estimated text 426 tok; estimated non-text 1269
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16736 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing terms:
  Clouds, High Street, Rural, Tree, Trees
- _Tokens:_ prompt 16736 tok; estimated text 426 tok; estimated non-text 16310
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=97%; missing terms: Colchester,
  Entrance, Essex, High Street, Path; keywords=19
- _Tokens:_ prompt 16752 tok; estimated text 426 tok; estimated non-text 16326
  tok; generated 116 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16736 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing terms:
  Entrance, High Street, Path, Rural, Tranquil
- _Tokens:_ prompt 16736 tok; estimated text 426 tok; estimated non-text 16310
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.59%; nontext prompt burden=97%; missing
  terms: Colchester, Essex, High Street, Rural, Tranquil; nonvisual metadata
  reused
- _Tokens:_ prompt 16752 tok; estimated text 426 tok; estimated non-text 16326
  tok; generated 99 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.70%; nontext prompt burden=97%; missing
  terms: High Street, Path, Tranquil, exterior, St; nonvisual metadata reused
- _Tokens:_ prompt 16752 tok; estimated text 426 tok; estimated non-text 16326
  tok; generated 118 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=97%; missing terms: Colchester,
  Entrance, Essex, High Street, Path
- _Tokens:_ prompt 16752 tok; estimated text 426 tok; estimated non-text 16326
  tok; generated 108 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.59%; nontext prompt burden=97%; missing
  terms: Entrance, Essex, High Street, Rural, Tranquil; nonvisual metadata
  reused
- _Tokens:_ prompt 16752 tok; estimated text 426 tok; estimated non-text 16326
  tok; generated 99 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.75%; nontext prompt burden=97%; missing
  terms: Entrance, Essex, High Street, Path, Rural; nonvisual metadata reused
- _Tokens:_ prompt 16752 tok; estimated text 426 tok; estimated non-text 16326
  tok; generated 126 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=72%; missing
  sections: title; missing terms: Clouds, Entrance, Essex, High Street, Path
- _Tokens:_ prompt 1502 tok; estimated text 426 tok; estimated non-text 1076
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.73%; nontext prompt burden=97%; missing
  terms: Colchester, Entrance, Essex, High Street, Path; nonvisual metadata
  reused
- _Tokens:_ prompt 16752 tok; estimated text 426 tok; estimated non-text 16326
  tok; generated 123 tok; requested max 500 tok; stop reason completed

