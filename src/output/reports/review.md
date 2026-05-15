# Automated Review Digest

_Generated on 2026-05-15 12:25:02 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 60 | Keywords 0 | Δ-75 | 29.9 tps | context ignored, degeneration, harness, missing sections
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (5/100) | Desc 41 | Keywords 0 | Δ-70 | 304.6 tps | context ignored, harness
- `mlx-community/Qwen3.6-27B-mxfp8`: ❌ F (5/100) | Desc 43 | Keywords 0 | Δ-70 | 17.9 tps | context ignored, cutoff, degeneration, harness, long context, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (16/100) | Desc 45 | Keywords 42 | Δ-59 | 33.4 tps | context ignored, harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (16/100) | Desc 45 | Keywords 42 | Δ-59 | 5.9 tps | context ignored, harness

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

- None — no models produced clean output meeting all quality thresholds for this prompt.

### `caveat`

| Model                              | Verdict          | Hint Handling                                                                     | Key Evidence                                                                                                                                                   |
|------------------------------------|------------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/GLM-4.6V-nvfp4`     | `context_budget` | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush | output/prompt=0.43% \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush |
| `mlx-community/MolmoPoint-8B-fp16` | `context_budget` | preserves trusted hints \| nonvisual metadata reused                              | output/prompt=4.14% \| nontext prompt burden=86% \| missing terms: Bench, rises \| keywords=19                                                                 |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                                  | Key Evidence                                                                                                                                                                                            |
|---------------------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `facebook/pe-av-large`                                  | `runtime_failure`   | not evaluated                                                                                                  | model error \| mlx lm model load model                                                                                                                                                                  |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `runtime_failure`   | not evaluated                                                                                                  | model error \| mlx model load model                                                                                                                                                                     |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `runtime_failure`   | not evaluated                                                                                                  | weight mismatch \| mlx model load weight mismatch                                                                                                                                                       |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | missing sections: keywords \| missing terms: Bench, Building, Bush, Clock tower, Clouds \| nonvisual metadata reused                                                                                    |
| `qnguyen3/nanoLLaVA`                                    | `model_shortcoming` | preserves trusted hints                                                                                        | missing sections: keywords \| missing terms: Bench, Building, Bush, Clock tower, Clouds                                                                                                                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | `harness`           | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | Output appears truncated to about 9 tokens. \| missing terms: Architecture, Bench, Bird, Building, Bush                                                                                                 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush \| degeneration=character_loop: 'orm' repeated                        |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `harness`           | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing terms: Architecture, Bench, Bird, Building, Bush |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush \| degeneration=character_loop: '1.' repeated                         |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `harness`           | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=70% \| missing terms: Architecture, Bench, Bird, Building, Bush |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush \| repetitive token=phrase: "3- 3- 3- 3-..."                          |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | nontext prompt burden=70% \| missing sections: title, keywords \| missing terms: Bench, Building, Bush, Churchyard, Clock tower \| nonvisual metadata reused                                            |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush \| repetitive token=phrase: "they, they, they, they,..."              |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush \| formatting=Unknown tags: &lt;row_1_col_1&gt;                       |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=66% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | nontext prompt burden=69% \| missing sections: title \| missing terms: Building, Bush, Gothic Revival architecture, Landscape, tranquil \| keyword duplication=39%                                      |
| `microsoft/Phi-3.5-vision-instruct`                     | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=66% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush \| repetitive token=•                                                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=84% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/InternVL3-8B-bf16`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=88% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=83% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush \| nonvisual metadata reused                                          |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/pixtral-12b-8bit`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/gemma-4-31b-it-4bit`                     | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush \| degeneration=character_loop: '_C' repeated                         |
| `mlx-community/InternVL3-14B-8bit`                      | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=88% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 323 occurrences). \| hit token cap (500) \| nontext prompt burden=84% \| missing sections: title, description, keywords      |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                           | nontext prompt burden=69% \| missing sections: title \| missing terms: Bench, Building, Bush, Dorking, Gothic Revival architecture \| nonvisual metadata reused                                         |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Bush, Church                                | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Bush, Church                                            |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush \| degeneration=repeated_punctuation: '##########...'                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | `harness`           | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| At long prompt length (16851 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97%          |
| `mlx-community/pixtral-12b-bf16`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                          |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                                                       |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Bush, Church                                | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Bush, Church                                            |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | At long prompt length (16866 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                   |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | At long prompt length (16866 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                   |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush \| nonvisual metadata reused | At long prompt length (16851 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                   |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | At long prompt length (16866 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                   |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush \| nonvisual metadata reused | At long prompt length (16866 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                   |
| `mlx-community/gemma-4-31b-bf16`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush                                                                       |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | At long prompt length (16866 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                   |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Bench, Bird, Building, Bush \| degeneration=repeated_punctuation: '##########...'                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | At long prompt length (16866 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                   |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Bench, Bird, Building, Bush                              | At long prompt length (16866 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords             |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                                           | Affected Models                                            | Issue Draft                                                                                                                              | Evidence Bundle   | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                                                               | 1: `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)             | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx`                                                    | Weight/config mismatch during model load              | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                                           | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx_mlx-model-load-weight-mismatch_001.md)   | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-lm`                                                 | Missing module/import during model load               | Model Error \| phase model_load \| ModuleNotFoundError                                                                                                                                                      | 1: `facebook/pe-av-large`                                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-lm_mlx-lm-model-load-model_001.md)       | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers              | 323 BPE space markers found in decoded text \| prompt=2,899 \| output/prompt=17.25% \| nontext burden=84% \| stop=max_tokens \| hit token cap (500)                                                         | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_encoding_001.md)                     | -                 | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;\|endoftext\|&gt; \| prompt_tokens=16851, repetitive output \| prompt=16,851 \| output/prompt=2.97% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | 1: `mlx-community/X-Reasoner-7B-8bit`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm_stop-token_001.md)                   | -                 | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~9 \| prompt=571 \| output/prompt=1.58% \| nontext burden=16% \| stop=completed \| 3 model cluster                                                                                          | 3: `mlx-community/FastVLM-0.5B-bf16` (+2)                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_model-config-mlx-vlm_prompt-template_001.md) | -                 | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | prompt_tokens=16851, repetitive output \| prompt=16,851 \| output/prompt=2.97% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) \| 8 model cluster                                           | 8: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+7)          | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_007_mlx-vlm-mlx_long-context_001.md)             | -                 | Full and reduced reruns avoid context collapse.           |

## Model Verdicts

### `facebook/pe-av-large`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx-lm`; reported package `mlx-lm`; failure stage
  `Model Error`; diagnostic code `MLX_LM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect the import path and installed package version that owns
  the missing module before treating this as a model failure.
- _Key signals:_ model error; mlx lm model load model
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


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


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bench, Building,
  Bush, Clock tower, Clouds; nonvisual metadata reused
- _Tokens:_ prompt 567 tok; estimated text 478 tok; estimated non-text 89 tok;
  generated 67 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bench, Building,
  Bush, Clock tower, Clouds
- _Tokens:_ prompt 567 tok; estimated text 478 tok; estimated non-text 89 tok;
  generated 82 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 9 tokens.; missing terms:
  Architecture, Bench, Bird, Building, Bush
- _Tokens:_ prompt 571 tok; estimated text 478 tok; estimated non-text 93 tok;
  generated 9 tok; requested max 500 tok; stop reason completed


### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=character_loop: 'orm' repeated
- _Tokens:_ prompt 828 tok; estimated text 478 tok; estimated non-text 350
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (0.5%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=70%; missing terms: Architecture, Bench, Bird, Building, Bush
- _Tokens:_ prompt 1584 tok; estimated text 478 tok; estimated non-text 1106
  tok; generated 8 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=character_loop: '1.' repeated
- _Tokens:_ prompt 828 tok; estimated text 478 tok; estimated non-text 350
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (0.5%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=70%; missing terms: Architecture, Bench, Bird, Building, Bush
- _Tokens:_ prompt 1584 tok; estimated text 478 tok; estimated non-text 1106
  tok; generated 8 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 3396 tok; estimated text 478 tok; estimated non-text 2918
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 1772 tok; estimated text 478 tok; estimated non-text 1294
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  repetitive token=phrase: "3- 3- 3- 3-..."
- _Tokens:_ prompt 843 tok; estimated text 478 tok; estimated non-text 365
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title, keywords;
  missing terms: Bench, Building, Bush, Churchyard, Clock tower; nonvisual
  metadata reused
- _Tokens:_ prompt 1584 tok; estimated text 478 tok; estimated non-text 1106
  tok; generated 100 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  repetitive token=phrase: "they, they, they, they,..."
- _Tokens:_ prompt 823 tok; estimated text 478 tok; estimated non-text 345
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  formatting=Unknown tags: &lt;row_1_col_1&gt;
- _Tokens:_ prompt 672 tok; estimated text 478 tok; estimated non-text 194
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 1772 tok; estimated text 478 tok; estimated non-text 1294
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=66%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 1394 tok; estimated text 478 tok; estimated non-text 916
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=69%; missing sections: title; missing
  terms: Building, Bush, Gothic Revival architecture, Landscape, tranquil;
  keyword duplication=39%
- _Tokens:_ prompt 1542 tok; estimated text 478 tok; estimated non-text 1064
  tok; generated 447 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=66%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 1394 tok; estimated text 478 tok; estimated non-text 916
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  repetitive token=•
- _Tokens:_ prompt 831 tok; estimated text 478 tok; estimated non-text 353
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 3397 tok; estimated text 478 tok; estimated non-text 2919
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=84%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 2992 tok; estimated text 478 tok; estimated non-text 2514
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 3397 tok; estimated text 478 tok; estimated non-text 2919
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 4656 tok; estimated text 478 tok; estimated non-text 4178
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=74%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 1872 tok; estimated text 478 tok; estimated non-text 1394
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 6568 tok; estimated text 478 tok; estimated non-text 6090
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=88%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 3886 tok; estimated text 478 tok; estimated non-text 3408
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=83%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 2850 tok; estimated text 478 tok; estimated non-text 2372
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 93% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.43%; nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 6568 tok; estimated text 478 tok; estimated non-text 6090
  tok; generated 28 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  nonvisual metadata reused
- _Tokens:_ prompt 832 tok; estimated text 478 tok; estimated non-text 354
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 6568 tok; estimated text 478 tok; estimated non-text 6090
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 3690 tok; estimated text 478 tok; estimated non-text 3212
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=character_loop: '_C' repeated
- _Tokens:_ prompt 843 tok; estimated text 478 tok; estimated non-text 365
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=88%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 3886 tok; estimated text 478 tok; estimated non-text 3408
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 3781 tok; estimated text 478 tok; estimated non-text 3303
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 323 occurrences).; hit token cap (500); nontext prompt
  burden=84%; missing sections: title, description, keywords
- _Tokens:_ prompt 2899 tok; estimated text 478 tok; estimated non-text 2421
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=69%; missing sections: title; missing
  terms: Bench, Building, Bush, Dorking, Gothic Revival architecture;
  nonvisual metadata reused
- _Tokens:_ prompt 1542 tok; estimated text 478 tok; estimated non-text 1064
  tok; generated 98 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Bush, Church
- _Tokens:_ prompt 1754 tok; estimated text 478 tok; estimated non-text 1276
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=repeated_punctuation: '##########...'
- _Tokens:_ prompt 538 tok; estimated text 478 tok; estimated non-text 60 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; At long prompt length (16851 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%
- _Tokens:_ prompt 16851 tok; estimated text 478 tok; estimated non-text 16373
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ output/prompt=4.14%; nontext prompt burden=86%; missing
  terms: Bench, rises; keywords=19
- _Tokens:_ prompt 3382 tok; estimated text 478 tok; estimated non-text 2904
  tok; generated 140 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Tokens:_ prompt 3690 tok; estimated text 478 tok; estimated non-text 3212
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush
- _Tokens:_ prompt 832 tok; estimated text 478 tok; estimated non-text 354
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Bush, Church
- _Tokens:_ prompt 1754 tok; estimated text 478 tok; estimated non-text 1276
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16851 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16851 tok; estimated text 478 tok; estimated non-text 16373
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush
- _Tokens:_ prompt 831 tok; estimated text 478 tok; estimated non-text 353
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=repeated_punctuation: '##########...'
- _Tokens:_ prompt 539 tok; estimated text 478 tok; estimated non-text 61 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=97%; missing sections: title, description, keywords
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

