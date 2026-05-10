# Automated Review Digest

_Generated on 2026-05-10 01:31:28 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/gemma-3-27b-it-qat-4bit`: 🏆 A (88/100) | Desc 98 | Keywords 85 | Δ+12 | 30.4 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (86/100) | Desc 93 | Keywords 79 | Δ+10 | 62.6 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (83/100) | Desc 95 | Keywords 92 | Δ+8 | 66.5 tps
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🏆 A (82/100) | Desc 100 | Keywords 84 | Δ+7 | 17.6 tps
- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (80/100) | Desc 100 | Keywords 77 | Δ+5 | 108.3 tps

### Watchlist

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) | Desc 44 | Keywords 0 | Δ-70 | 241.5 tps | context ignored, harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | Δ-69 | 31.9 tps | harness, metadata borrowing, missing sections
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (9/100) | Desc 48 | Keywords 0 | Δ-66 | 65.6 tps | context ignored, harness, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (35/100) | Desc 73 | Keywords 69 | Δ-40 | 5.9 tps | context ignored, harness
- `microsoft/Phi-3.5-vision-instruct`: 🟡 C (64/100) | Desc 100 | Keywords 61 | Δ-11 | 52.0 tps | harness, metadata borrowing

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict   | Hint Handling           | Key Evidence                                                                                        |
|-----------------------------------------------------|-----------|-------------------------|-----------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`   | preserves trusted hints | missing terms: Chapel, Cross, Dorking, Objects, Station wagon                                       |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`   | preserves trusted hints | nontext prompt burden=85% \| missing terms: Bell Tower, Chapel, Cross, Dorking, Fence               |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`   | preserves trusted hints | nontext prompt burden=85% \| missing terms: Bell Tower, Chapel, Cross, Dorking, Gothic Architecture |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `clean`   | preserves trusted hints | missing terms: Bell Tower, Chapel, Cross, Dorking, Fence                                            |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `clean`   | preserves trusted hints | missing terms: Chapel, Cross, Dorking, Objects, Station wagon                                       |

### `caveat`

| Model                                             | Verdict          | Hint Handling                                        | Key Evidence                                                                                                                                              |
|---------------------------------------------------|------------------|------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=4.95% \| nontext prompt burden=85% \| missing terms: Bell Tower, Chapel, Dorking, Gothic Architecture, Objects \| nonvisual metadata reused |
| `mlx-community/InternVL3-8B-bf16`                 | `clean`          | preserves trusted hints                              | nontext prompt burden=80% \| missing terms: Chapel, Dorking, Objects, Surrey, low                                                                         |
| `mlx-community/pixtral-12b-8bit`                  | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=3.51% \| nontext prompt burden=86% \| missing terms: Bell Tower, Chapel, Dorking, Objects, Surrey \| nonvisual metadata reused              |
| `mlx-community/pixtral-12b-bf16`                  | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=3.48% \| nontext prompt burden=86% \| missing terms: Bell Tower, Chapel, Dorking, Objects, Station wagon \| nonvisual metadata reused       |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                                  | Key Evidence                                                                                                                                                                                                                                             |
|---------------------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `runtime_failure`   | not evaluated                                                                                                  | model error \| mlx model load model                                                                                                                                                                                                                      |
| `facebook/pe-av-large`                                  | `runtime_failure`   | not evaluated                                                                                                  | model error \| mlx vlm model load model                                                                                                                                                                                                                  |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | keywords=58 \| context echo=100% \| nonvisual metadata reused \| reasoning leak                                                                                                                                                                          |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `runtime_failure`   | not evaluated                                                                                                  | model error \| mlx model load model                                                                                                                                                                                                                      |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                           | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=57 \| context echo=100% \| nonvisual metadata reused                                                                                                                    |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                           | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=57 \| context echo=100% \| nonvisual metadata reused                                                                                                                    |
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure`   | not evaluated                                                                                                  | processor error \| model config processor load processor                                                                                                                                                                                                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | missing sections: keywords \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking \| nonvisual metadata reused                                                                                                                                   |
| `qnguyen3/nanoLLaVA`                                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | missing sections: keywords \| missing terms: Bell Tower, Blue sky, Chapel, Cross, Daylight \| nonvisual metadata reused                                                                                                                                  |
| `mlx-community/FastVLM-0.5B-bf16`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | missing sections: keywords \| missing terms: sunny, day \| nonvisual metadata reused                                                                                                                                                                     |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `model_shortcoming` | ignores trusted hints \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                              | nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                                                                                                                  |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | keywords=20 \| context echo=53% \| nonvisual metadata reused                                                                                                                                                                                             |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `model_shortcoming` | ignores trusted hints \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                              | nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                                                                                                                  |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | nontext prompt burden=73% \| keywords=20 \| nonvisual metadata reused                                                                                                                                                                                    |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | nontext prompt burden=73% \| keywords=20 \| nonvisual metadata reused                                                                                                                                                                                    |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | nontext prompt burden=66% \| missing terms: Bell Tower, Chapel, Cross, Fence, Objects \| keywords=19 \| nonvisual metadata reused                                                                                                                        |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `harness`           | ignores trusted hints \| missing terms: Bell Tower, Blue sky, Car, Chapel, Cross                               | Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=83% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Blue sky, Car, Chapel, Cross |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `harness`           | ignores trusted hints \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                              | Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                                                  |
| `mlx-community/InternVL3-14B-8bit`                      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | nontext prompt burden=80% \| missing terms: Dorking, Station wagon, Surrey, low, angle \| nonvisual metadata reused                                                                                                                                      |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church \| nonvisual metadata reused                                                                                           |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | nontext prompt burden=83% \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking \| context echo=47% \| nonvisual metadata reused                                                                                                                |
| `mlx-community/gemma-4-31b-it-4bit`                     | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | missing terms: Chapel, Cross, Dorking, Objects, Surrey \| nonvisual metadata reused                                                                                                                                                                      |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                           | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 57 occurrences). \| nontext prompt burden=82% \| missing sections: description, keywords \| missing terms: Dorking, Gothic Architecture, Objects, Surrey, low                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | missing sections: title, description, keywords \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking \| nonvisual metadata reused                                                                                                               |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                           | hit token cap (500) \| nontext prompt burden=69% \| missing sections: title, description \| missing terms: Bell Tower, Gothic Architecture, Steeple, Surrey, pictured                                                                                    |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | missing sections: title, description, keywords \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking \| nonvisual metadata reused                                                                                                               |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`           | preserves trusted hints \| nonvisual metadata reused                                                           | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=66%                                                         |
| `mlx-community/gemma-4-31b-bf16`                        | `model_shortcoming` | ignores trusted hints \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                              | missing sections: title, description, keywords \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                                                                                                                                               |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | missing sections: title, description, keywords \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking \| nonvisual metadata reused                                                                                                               |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                                                                                           |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                           | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Blue sky, Chapel, Cross, Daylight                                                                                       |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                           | hit token cap (500) \| nontext prompt burden=86% \| missing sections: description, keywords \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking                                                                                               |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | preserves trusted hints                                                                                        | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking                                                                                        |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                              | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                                                                                           |
| `mlx-community/X-Reasoner-7B-8bit`                      | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                           | hit token cap (500) \| nontext prompt burden=97% \| missing terms: Bell Tower, Chapel, wide, shot, sunny \| keyword duplication=74%                                                                                                                      |
| `mlx-community/GLM-4.6V-nvfp4`                          | `cutoff_degraded`   | preserves trusted hints                                                                                        | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking                                                                                                  |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                              | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                                                                                           |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness`           | ignores trusted hints \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                              | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| Output appears truncated to about 5 tokens. \| nontext prompt burden=97% \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church                                          |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | preserves trusted hints                                                                                        | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Chapel, Cross, Dorking, Gothic Architecture                                                                             |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                           | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Chapel, Cross, Daylight, Gothic Architecture, Objects                                                                               |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                           | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Chapel, Daylight, Dorking, Gothic Architecture                                                                          |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | preserves trusted hints                                                                                        | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking                                                                                        |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                           | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking                                                                                        |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints                                                                                        | hit token cap (500) \| nontext prompt burden=97% \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking \| keywords=32                                                                                                                           |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                           | hit token cap (500) \| nontext prompt burden=69% \| missing sections: title, description \| missing terms: Bell Tower, Daylight, Gothic Architecture, Steeple, its                                                                                       |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                           | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: Bell Tower, Chapel, Cross, Daylight, Dorking                                                                                        |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

| Target                                                   | Problem                                                                                              | Evidence Snapshot                                                                                                                                                                                                        | Affected Models                                            | Issue Draft                                                                                                                                                              | Evidence Bundle   | Fixed When                                                |
|----------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load                                                             | Model Error \| phase model_load \| ValueError \| 2 model cluster                                                                                                                                                         | 2: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (+1)                 | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)                                             | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | mlx-vlm: Decode / runtime error: property 'text' of 'NaiveStreamingDetokenizer' object has no setter | Error \| phase decode \| AttributeError \| 3 model cluster                                                                                                                                                               | 3: `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (+2) | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_mlx-vlm-decode-error_001.md)                                         | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Missing module/import during model load                                                              | Model Error \| phase model_load \| ValueError                                                                                                                                                                            | 1: `facebook/pe-av-large`                                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm_mlx-vlm-model-load-model_001.md)                                     | -                 | Load/generation completes or fails with a narrower owner. |
| model configuration / repository                         | Processor config is missing image processor                                                          | Processor Error \| phase processor_load \| ValueError                                                                                                                                                                    | 1: `mlx-community/MolmoPoint-8B-fp16`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_model-configuration-repository_model-config-processor-load-processor_001.md) | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers                                                             | 57 BPE space markers found in decoded text \| prompt=2,676 \| output/prompt=3.48% \| nontext burden=82% \| stop=completed                                                                                                | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm_encoding_001.md)                                                     | -                 | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text                                                       | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=1,387 \| output/prompt=36.05% \| nontext burden=66% \| stop=max_tokens \| hit token cap (500) | 1: `microsoft/Phi-3.5-vision-instruct`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm_stop-token_001.md)                                                   | -                 | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                                                                | output/prompt=0.4% \| prompt=2,789 \| output/prompt=0.39% \| nontext burden=83% \| stop=completed \| 2 model cluster                                                                                                     | 2: `mlx-community/llava-v1.6-mistral-7b-8bit` (+1)         | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_007_model-config-mlx-vlm_prompt-template_001.md)                                 | -                 | Requested sections render without template leakage.       |
| model repo first; mlx-vlm if template handling disagrees | Stop/control tokens leaked into generated text                                                       | decoded text contains control token &lt;\|endoftext\|&gt; \| generated_tokens~5 \| prompt=16,789 \| output/prompt=0.03% \| nontext burden=97% \| stop=completed                                                          | 1: `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_008_model-config-mlx-vlm_stop-token_001.md)                                      | -                 | No leaked stop/control tokens.                            |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short                                                | token cap \| missing sections \| abrupt tail \| prompt=16,804 \| output/prompt=2.98% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500)                                                                     | 1: `mlx-community/Qwen3.5-9B-MLX-4bit`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_009_mlx-vlm-mlx_long-context_001.md)                                             | -                 | Full and reduced reruns avoid context collapse.           |

## Model Verdicts

### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Model
  Error`; diagnostic code `MLX_MODEL_LOAD_MODEL`
- _Next step:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.
- _Key signals:_ model error; mlx model load model
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `facebook/pe-av-large`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx-vlm`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect the import path and installed package version that owns
  the missing module before treating this as a model failure.
- _Key signals:_ model error; mlx vlm model load model
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`; reported package `mlx-vlm`; failure stage
  `Error`; diagnostic code `MLX_VLM_DECODE_ERROR`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=58; context echo=100%; nonvisual metadata reused;
  reasoning leak
- _Tokens:_ prompt n/a; estimated text 474 tok; estimated non-text n/a;
  generated n/a; requested max 500 tok; stop reason exception


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Model
  Error`; diagnostic code `MLX_MODEL_LOAD_MODEL`
- _Next step:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.
- _Key signals:_ model error; mlx model load model
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`; reported
  package `mlx-vlm`; failure stage `Error`; diagnostic code
  `MLX_VLM_DECODE_ERROR`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|im_end|&gt; appeared in generated
  text.; keywords=57; context echo=100%; nonvisual metadata reused
- _Tokens:_ prompt n/a; estimated text 474 tok; estimated non-text n/a;
  generated n/a; requested max 500 tok; stop reason exception


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`; reported
  package `mlx-vlm`; failure stage `Error`; diagnostic code
  `MLX_VLM_DECODE_ERROR`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|im_end|&gt; appeared in generated
  text.; keywords=57; context echo=100%; nonvisual metadata reused
- _Tokens:_ prompt n/a; estimated text 474 tok; estimated non-text n/a;
  generated n/a; requested max 500 tok; stop reason exception


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `model-config`; reported package `model-config`;
  failure stage `Processor Error`; diagnostic code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
- _Next step:_ Inspect the model repo processor/preprocessor config and
  AutoProcessor mapping; the multimodal processor is missing or not exposing
  the image processor expected by mlx-vlm.
- _Key signals:_ processor error; model config processor load processor
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bell Tower,
  Chapel, Cross, Daylight, Dorking; nonvisual metadata reused
- _Tokens:_ prompt 561 tok; estimated text 474 tok; estimated non-text 87 tok;
  generated 137 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bell Tower, Blue
  sky, Chapel, Cross, Daylight; nonvisual metadata reused
- _Tokens:_ prompt 561 tok; estimated text 474 tok; estimated non-text 87 tok;
  generated 68 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: sunny, day;
  nonvisual metadata reused
- _Tokens:_ prompt 565 tok; estimated text 474 tok; estimated non-text 91 tok;
  generated 222 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Chapel, Cross, Dorking, Objects, Station wagon
- _Tokens:_ prompt 835 tok; estimated text 474 tok; estimated non-text 361
  tok; generated 86 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title,
  description, keywords; missing terms: Bell Tower, Blue sky, Car, Chapel,
  Church
- _Tokens:_ prompt 1585 tok; estimated text 474 tok; estimated non-text 1111
  tok; generated 17 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; context echo=53%; nonvisual metadata reused
- _Tokens:_ prompt 671 tok; estimated text 474 tok; estimated non-text 197
  tok; generated 131 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title,
  description, keywords; missing terms: Bell Tower, Blue sky, Car, Chapel,
  Church
- _Tokens:_ prompt 1585 tok; estimated text 474 tok; estimated non-text 1111
  tok; generated 20 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%; keywords=20; nonvisual metadata
  reused
- _Tokens:_ prompt 1771 tok; estimated text 474 tok; estimated non-text 1297
  tok; generated 182 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 85% and the output stays weak under that load.
- _Key signals:_ output/prompt=4.95%; nontext prompt burden=85%; missing
  terms: Bell Tower, Chapel, Dorking, Gothic Architecture, Objects; nonvisual
  metadata reused
- _Tokens:_ prompt 3172 tok; estimated text 474 tok; estimated non-text 2698
  tok; generated 157 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%; keywords=20; nonvisual metadata
  reused
- _Tokens:_ prompt 1771 tok; estimated text 474 tok; estimated non-text 1297
  tok; generated 182 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=66%; missing terms: Bell Tower, Chapel,
  Cross, Fence, Objects; keywords=19; nonvisual metadata reused
- _Tokens:_ prompt 1387 tok; estimated text 474 tok; estimated non-text 913
  tok; generated 129 tok; requested max 500 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Check chat-template and EOS defaults first; the output shape is
  not matching the requested contract.
- _Key signals:_ Output is very short relative to prompt size (0.4%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=83%; missing sections: title, description, keywords; missing terms:
  Bell Tower, Blue sky, Car, Chapel, Cross
- _Tokens:_ prompt 2789 tok; estimated text 474 tok; estimated non-text 2315
  tok; generated 11 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (0.6%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=70%; missing terms: Bell Tower, Blue sky, Car, Chapel, Church
- _Tokens:_ prompt 1585 tok; estimated text 474 tok; estimated non-text 1111
  tok; generated 9 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=80%; missing terms: Chapel, Dorking,
  Objects, Surrey, low
- _Tokens:_ prompt 2344 tok; estimated text 474 tok; estimated non-text 1870
  tok; generated 75 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=85%; missing terms: Bell Tower, Chapel,
  Cross, Dorking, Fence
- _Tokens:_ prompt 3173 tok; estimated text 474 tok; estimated non-text 2699
  tok; generated 108 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=85%; missing terms: Bell Tower, Chapel,
  Cross, Dorking, Gothic Architecture
- _Tokens:_ prompt 3173 tok; estimated text 474 tok; estimated non-text 2699
  tok; generated 116 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bell Tower, Chapel, Cross, Dorking, Fence
- _Tokens:_ prompt 830 tok; estimated text 474 tok; estimated non-text 356
  tok; generated 94 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=80%; missing terms: Dorking, Station
  wagon, Surrey, low, angle; nonvisual metadata reused
- _Tokens:_ prompt 2344 tok; estimated text 474 tok; estimated non-text 1870
  tok; generated 86 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bell Tower, Blue sky, Car, Chapel, Church;
  nonvisual metadata reused
- _Tokens:_ prompt 821 tok; estimated text 474 tok; estimated non-text 347
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=83%; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking; context echo=47%; nonvisual metadata reused
- _Tokens:_ prompt 2844 tok; estimated text 474 tok; estimated non-text 2370
  tok; generated 117 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Chapel, Cross, Dorking, Objects, Surrey;
  nonvisual metadata reused
- _Tokens:_ prompt 835 tok; estimated text 474 tok; estimated non-text 361
  tok; generated 94 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ output/prompt=3.51%; nontext prompt burden=86%; missing
  terms: Bell Tower, Chapel, Dorking, Objects, Surrey; nonvisual metadata
  reused
- _Tokens:_ prompt 3366 tok; estimated text 474 tok; estimated non-text 2892
  tok; generated 118 tok; requested max 500 tok; stop reason completed


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 57 occurrences).; nontext prompt burden=82%; missing sections:
  description, keywords; missing terms: Dorking, Gothic Architecture, Objects,
  Surrey, low
- _Tokens:_ prompt 2676 tok; estimated text 474 tok; estimated non-text 2202
  tok; generated 93 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bell Tower, Chapel, Cross, Daylight, Dorking; nonvisual metadata
  reused
- _Tokens:_ prompt 829 tok; estimated text 474 tok; estimated non-text 355
  tok; generated 316 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Chapel, Cross, Dorking, Objects, Station wagon
- _Tokens:_ prompt 830 tok; estimated text 474 tok; estimated non-text 356
  tok; generated 106 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ output/prompt=3.48%; nontext prompt burden=86%; missing
  terms: Bell Tower, Chapel, Dorking, Objects, Station wagon; nonvisual
  metadata reused
- _Tokens:_ prompt 3366 tok; estimated text 474 tok; estimated non-text 2892
  tok; generated 117 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=69%; missing
  sections: title, description; missing terms: Bell Tower, Gothic
  Architecture, Steeple, Surrey, pictured
- _Tokens:_ prompt 1553 tok; estimated text 474 tok; estimated non-text 1079
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bell Tower, Chapel, Cross, Daylight, Dorking; nonvisual metadata
  reused
- _Tokens:_ prompt 532 tok; estimated text 474 tok; estimated non-text 58 tok;
  generated 161 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; Special control token &lt;|endoftext|&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=66%
- _Tokens:_ prompt 1387 tok; estimated text 474 tok; estimated non-text 913
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bell Tower, Blue sky, Car, Chapel, Church
- _Tokens:_ prompt 823 tok; estimated text 474 tok; estimated non-text 349
  tok; generated 51 tok; requested max 500 tok; stop reason completed


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bell Tower, Chapel, Cross, Daylight, Dorking; nonvisual metadata
  reused
- _Tokens:_ prompt 533 tok; estimated text 474 tok; estimated non-text 59 tok;
  generated 50 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Blue sky,
  Car, Chapel, Church
- _Tokens:_ prompt 4657 tok; estimated text 474 tok; estimated non-text 4183
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Blue sky,
  Chapel, Cross, Daylight
- _Tokens:_ prompt 6675 tok; estimated text 474 tok; estimated non-text 6201
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: description, keywords; missing terms: Bell Tower, Chapel, Cross,
  Daylight, Dorking
- _Tokens:_ prompt 3457 tok; estimated text 474 tok; estimated non-text 2983
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking
- _Tokens:_ prompt 6675 tok; estimated text 474 tok; estimated non-text 6201
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Blue sky,
  Car, Chapel, Church
- _Tokens:_ prompt 1748 tok; estimated text 474 tok; estimated non-text 1274
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: Bell Tower, Chapel, wide, shot, sunny; keyword duplication=74%
- _Tokens:_ prompt 16789 tok; estimated text 474 tok; estimated non-text 16315
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description; missing terms: Bell Tower, Chapel, Cross,
  Daylight, Dorking
- _Tokens:_ prompt 6675 tok; estimated text 474 tok; estimated non-text 6201
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Blue sky,
  Car, Chapel, Church
- _Tokens:_ prompt 1748 tok; estimated text 474 tok; estimated non-text 1274
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; Output appears truncated to about 5 tokens.; nontext prompt
  burden=97%; missing terms: Bell Tower, Blue sky, Car, Chapel, Church
- _Tokens:_ prompt 16789 tok; estimated text 474 tok; estimated non-text 16315
  tok; generated 5 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Cross, Dorking, Gothic Architecture
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Chapel, Cross,
  Daylight, Gothic Architecture, Objects
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Daylight, Dorking, Gothic Architecture
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: Bell Tower, Chapel, Cross, Daylight, Dorking; keywords=32
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=69%; missing
  sections: title, description; missing terms: Bell Tower, Daylight, Gothic
  Architecture, Steeple, its
- _Tokens:_ prompt 1553 tok; estimated text 474 tok; estimated non-text 1079
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

