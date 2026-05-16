# Automated Review Digest

_Generated on 2026-05-16 00:16:45 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Watchlist

- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-49 | 245.4 tps | context ignored, harness
- `mlx-community/InternVL3-8B-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-49 | 49.6 tps | context ignored, harness, long context
- `mlx-community/InternVL3-14B-8bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-49 | 56.4 tps | context ignored, harness, long context
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-49 | 33.3 tps | context ignored, harness, long context
- `mlx-community/Qwen3.5-35B-A3B-6bit`: ❌ F (20/100) | Desc 60 | Keywords 0 | Δ-29 | 79.7 tps | context ignored, cutoff, degeneration, harness, long context, missing sections, repetitive

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

- None — no models produced clean output meeting all quality thresholds for this prompt.

### `caveat`

| Model                                                   | Verdict          | Hint Handling                                                                    | Key Evidence                                                                                                                                                                                  |
|---------------------------------------------------------|------------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/InternVL3-8B-bf16`                       | `context_budget` | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate | Output appears truncated to about 3 tokens. \| At long prompt length (3031 tokens), output stayed unusually short (3 tokens; ratio 0.1%). \| output/prompt=0.10% \| nontext prompt burden=86% |
| `mlx-community/InternVL3-14B-8bit`                      | `context_budget` | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate | Output appears truncated to about 2 tokens. \| At long prompt length (3031 tokens), output stayed unusually short (2 tokens; ratio 0.1%). \| output/prompt=0.07% \| nontext prompt burden=86% |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `context_budget` | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate | Output appears truncated to about 2 tokens. \| At long prompt length (3581 tokens), output stayed unusually short (2 tokens; ratio 0.1%). \| output/prompt=0.06% \| nontext prompt burden=88% |
| `mlx-community/GLM-4.6V-nvfp4`                          | `context_budget` | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate | output/prompt=0.29% \| nontext prompt burden=94% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                 |

### `needs_triage`

- None.

### `avoid`

| Model                                               | Verdict             | Hint Handling                                                                                                 | Key Evidence                                                                                                                                                                                                         |
|-----------------------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `facebook/pe-av-large`                              | `runtime_failure`   | not evaluated                                                                                                 | model error \| mlx lm model load model                                                                                                                                                                               |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`           | `runtime_failure`   | not evaluated                                                                                                 | model error \| mlx model load model                                                                                                                                                                                  |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                 | `runtime_failure`   | not evaluated                                                                                                 | weight mismatch \| mlx model load weight mismatch                                                                                                                                                                    |
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `model_shortcoming` | ignores trusted hints \| missing terms: Rochester, turns, celebrate, Medway, winning                          | missing sections: keywords \| missing terms: Rochester, turns, celebrate, Medway, winning                                                                                                                            |
| `mlx-community/FastVLM-0.5B-bf16`                   | `harness`           | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | Output appears truncated to about 4 tokens. \| missing terms: Rochester, Castle, turns, Red, celebrate                                                                                                               |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                  | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate \| nonvisual metadata reused                                                        |
| `qnguyen3/nanoLLaVA`                                | `model_shortcoming` | improves trusted hints                                                                                        | missing sections: keywords \| missing terms: turns, celebrate, Medway, winning, bid                                                                                                                                  |
| `mlx-community/LFM2-VL-1.6B-8bit`                   | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate \| repetitive token="                                                               |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | `model_shortcoming` | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                                               |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`  | `model_shortcoming` | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | nontext prompt burden=77% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                                               |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=80% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/gemma-3n-E2B-4bit`                   | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate \| repetitive token=they,                                                           |
| `mlx-community/SmolVLM-Instruct-bf16`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=80% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate \| formatting=Unknown tags: &lt;em&gt;, &lt;li&gt;                                  |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                           | nontext prompt burden=72% \| missing terms: turns, celebrate, winning, its, European \| nonvisual metadata reused \| reasoning leak                                                                                  |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`          | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate \| repetitive token=phrase: "outbre outbre outbre outbre..."                        |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | `model_shortcoming` | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate \| nonvisual metadata reused | nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate \| nonvisual metadata reused                                                  |
| `microsoft/Phi-3.5-vision-instruct`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=68% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=68% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/gemma-3n-E4B-it-bf16`                | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate \| repetitive token=1)                                                              |
| `mlx-community/llava-v1.6-mistral-7b-8bit`          | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=88% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/paligemma2-3b-pt-896-4bit`           | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=91% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=94% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`     | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=91% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/GLM-4.6V-Flash-6bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=94% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=88% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                                                     |
| `mlx-community/Molmo-7B-D-0924-8bit`                | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/gemma-4-31b-it-4bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                                                     |
| `mlx-community/pixtral-12b-8bit`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=91% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate \| degeneration=repeated_punctuation: '##########...'                               |
| `mlx-community/Molmo-7B-D-0924-bf16`                | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`     | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, turns, celebrate, Medway, winning                          | hit token cap (500) \| nontext prompt burden=73% \| missing sections: title, description, keywords \| missing terms: Rochester, turns, celebrate, Medway, winning                                                    |
| `mlx-community/X-Reasoner-7B-8bit`                  | `harness`           | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| At long prompt length (16868 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=98% |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                                                     |
| `mlx-community/MolmoPoint-8B-fp16`                  | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                          | nontext prompt burden=83% \| missing terms: turns, celebrate, winning, its, European \| nonvisual metadata reused                                                                                                    |
| `mlx-community/pixtral-12b-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| nontext prompt burden=91% \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                        |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | At long prompt length (16882 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=98% \| missing sections: title, description, keywords                                                |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | At long prompt length (16882 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=98% \| missing sections: title, description, keywords                                                |
| `mlx-community/gemma-4-31b-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate                                                                                     |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                           | nontext prompt burden=72% \| missing terms: turns, celebrate, Medway, winning, its \| nonvisual metadata reused \| reasoning leak                                                                                    |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | At long prompt length (16882 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=98% \| missing sections: title, description, keywords                                                |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`           | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | At long prompt length (16868 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=98% \| missing sections: title, description, keywords                          |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Rochester, Castle, turns, Red, celebrate \| degeneration=repeated_punctuation: '##########...'                               |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | At long prompt length (16882 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=98% \| missing sections: title, description, keywords                                                |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | At long prompt length (16882 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=98% \| missing sections: title, description, keywords                                                |
| `mlx-community/Qwen3.5-27B-4bit`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate \| nonvisual metadata reused | At long prompt length (16882 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=98% \| missing sections: title, description, keywords                          |
| `mlx-community/Qwen3.6-27B-mxfp8`                   | `cutoff_degraded`   | ignores trusted hints \| missing terms: Rochester, Castle, turns, Red, celebrate                              | At long prompt length (16882 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=98% \| missing sections: title, description, keywords                                                |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                                                      | Affected Models                                                 | Issue Draft                                                                                                                              | Evidence Bundle   | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                                                                          | 1: `mlx-community/Kimi-VL-A3B-Thinking-8bit`                    | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)             | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx`                                                    | Weight/config mismatch during model load              | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                                                      | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                          | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx_mlx-model-load-weight-mismatch_001.md)   | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-lm`                                                 | Missing module/import during model load               | Model Error \| phase model_load \| ModuleNotFoundError                                                                                                                                                                 | 1: `facebook/pe-av-large`                                       | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-lm_mlx-lm-model-load-model_001.md)       | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;\|endoftext\|&gt; \| prompt_tokens=16868, prompt/image context dropped \| prompt=16,868 \| output/prompt=2.96% \| nontext burden=98% \| stop=max_tokens \| hit token cap (500) | 1: `mlx-community/X-Reasoner-7B-8bit`                           | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_stop-token_001.md)                   | -                 | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~4 \| prompt=484 \| output/prompt=0.83% \| nontext burden=14% \| stop=completed                                                                                                                        | 1: `mlx-community/FastVLM-0.5B-bf16`                            | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_model-config-mlx-vlm_prompt-template_001.md) | -                 | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | generated_tokens~2 \| prompt_tokens=3581, output_tokens=2, output/prompt=0.1% \| prompt=3,581 \| output/prompt=0.06% \| nontext burden=88% \| stop=completed \| 3 model cluster                                        | 3: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (+2) | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm-mlx_long-context_001.md)             | -                 | Full and reduced reruns avoid context collapse.           |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | prompt_tokens=16868, prompt/image context dropped \| prompt=16,868 \| output/prompt=2.96% \| nontext burden=98% \| stop=max_tokens \| hit token cap (500) \| 8 model cluster                                           | 8: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+7)               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_007_mlx-vlm-mlx_long-context_002.md)             | -                 | Full and reduced reruns avoid context collapse.           |

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
- _Key signals:_ missing sections: keywords; missing terms: Rochester, turns,
  celebrate, Medway, winning
- _Tokens:_ prompt 480 tok; estimated text 414 tok; estimated non-text 66 tok;
  generated 38 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 4 tokens.; missing terms:
  Rochester, Castle, turns, Red, celebrate
- _Tokens:_ prompt 484 tok; estimated text 414 tok; estimated non-text 70 tok;
  generated 4 tok; requested max 500 tok; stop reason completed


### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate; nonvisual
  metadata reused
- _Tokens:_ prompt 727 tok; estimated text 414 tok; estimated non-text 313
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: turns, celebrate,
  Medway, winning, bid
- _Tokens:_ prompt 480 tok; estimated text 414 tok; estimated non-text 66 tok;
  generated 105 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 3 tokens.; At long prompt
  length (3031 tokens), output stayed unusually short (3 tokens; ratio 0.1%).;
  output/prompt=0.10%; nontext prompt burden=86%
- _Tokens:_ prompt 3031 tok; estimated text 414 tok; estimated non-text 2617
  tok; generated 3 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  repetitive token="
- _Tokens:_ prompt 727 tok; estimated text 414 tok; estimated non-text 313
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=73%; missing sections: title,
  description, keywords; missing terms: Rochester, Castle, turns, Red,
  celebrate
- _Tokens:_ prompt 1508 tok; estimated text 414 tok; estimated non-text 1094
  tok; generated 24 tok; requested max 500 tok; stop reason completed


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=77%; missing sections: title,
  description, keywords; missing terms: Rochester, Castle, turns, Red,
  celebrate
- _Tokens:_ prompt 1770 tok; estimated text 414 tok; estimated non-text 1356
  tok; generated 64 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 2 tokens.; At long prompt
  length (3031 tokens), output stayed unusually short (2 tokens; ratio 0.1%).;
  output/prompt=0.07%; nontext prompt burden=86%
- _Tokens:_ prompt 3031 tok; estimated text 414 tok; estimated non-text 2617
  tok; generated 2 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=80%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 2055 tok; estimated text 414 tok; estimated non-text 1641
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  repetitive token=they,
- _Tokens:_ prompt 743 tok; estimated text 414 tok; estimated non-text 329
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=80%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 2055 tok; estimated text 414 tok; estimated non-text 1641
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  formatting=Unknown tags: &lt;em&gt;, &lt;li&gt;
- _Tokens:_ prompt 753 tok; estimated text 414 tok; estimated non-text 339
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=72%; missing terms: turns, celebrate,
  winning, its, European; nonvisual metadata reused; reasoning leak
- _Tokens:_ prompt 1480 tok; estimated text 414 tok; estimated non-text 1066
  tok; generated 299 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 4079 tok; estimated text 414 tok; estimated non-text 3665
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 88% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 2 tokens.; At long prompt
  length (3581 tokens), output stayed unusually short (2 tokens; ratio 0.1%).;
  output/prompt=0.06%; nontext prompt burden=88%
- _Tokens:_ prompt 3581 tok; estimated text 414 tok; estimated non-text 3167
  tok; generated 2 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  repetitive token=phrase: "outbre outbre outbre outbre..."
- _Tokens:_ prompt 591 tok; estimated text 414 tok; estimated non-text 177
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=73%; missing sections: title,
  description, keywords; missing terms: Rochester, Castle, turns, Red,
  celebrate; nonvisual metadata reused
- _Tokens:_ prompt 1508 tok; estimated text 414 tok; estimated non-text 1094
  tok; generated 39 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=68%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 1290 tok; estimated text 414 tok; estimated non-text 876
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=68%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 1290 tok; estimated text 414 tok; estimated non-text 876
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  repetitive token=1)
- _Tokens:_ prompt 751 tok; estimated text 414 tok; estimated non-text 337
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=88%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 3460 tok; estimated text 414 tok; estimated non-text 3046
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 4580 tok; estimated text 414 tok; estimated non-text 4166
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 4080 tok; estimated text 414 tok; estimated non-text 3666
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 4080 tok; estimated text 414 tok; estimated non-text 3666
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=94%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 6535 tok; estimated text 414 tok; estimated non-text 6121
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 4718 tok; estimated text 414 tok; estimated non-text 4304
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=94%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 6535 tok; estimated text 414 tok; estimated non-text 6121
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=88%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 3472 tok; estimated text 414 tok; estimated non-text 3058
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 94% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.29%; nontext prompt burden=94%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 6535 tok; estimated text 414 tok; estimated non-text 6121
  tok; generated 19 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate
- _Tokens:_ prompt 752 tok; estimated text 414 tok; estimated non-text 338
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=71%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 1435 tok; estimated text 414 tok; estimated non-text 1021
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate
- _Tokens:_ prompt 753 tok; estimated text 414 tok; estimated non-text 339
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 4627 tok; estimated text 414 tok; estimated non-text 4213
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  degeneration=repeated_punctuation: '##########...'
- _Tokens:_ prompt 448 tok; estimated text 414 tok; estimated non-text 34 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=71%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 1435 tok; estimated text 414 tok; estimated non-text 1021
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Rochester, turns,
  celebrate, Medway, winning
- _Tokens:_ prompt 1508 tok; estimated text 414 tok; estimated non-text 1094
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; At long prompt length (16868 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=98%
- _Tokens:_ prompt 16868 tok; estimated text 414 tok; estimated non-text 16454
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate
- _Tokens:_ prompt 752 tok; estimated text 414 tok; estimated non-text 338
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=83%; missing terms: turns, celebrate,
  winning, its, European; nonvisual metadata reused
- _Tokens:_ prompt 2478 tok; estimated text 414 tok; estimated non-text 2064
  tok; generated 196 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Tokens:_ prompt 4627 tok; estimated text 414 tok; estimated non-text 4213
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate
- _Tokens:_ prompt 741 tok; estimated text 414 tok; estimated non-text 327
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=72%; missing terms: turns, celebrate,
  Medway, winning, its; nonvisual metadata reused; reasoning leak
- _Tokens:_ prompt 1480 tok; estimated text 414 tok; estimated non-text 1066
  tok; generated 300 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16868 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=98%; missing sections: title, description, keywords
- _Tokens:_ prompt 16868 tok; estimated text 414 tok; estimated non-text 16454
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  degeneration=repeated_punctuation: '##########...'
- _Tokens:_ prompt 449 tok; estimated text 414 tok; estimated non-text 35 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=98%; missing sections: title, description, keywords
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

