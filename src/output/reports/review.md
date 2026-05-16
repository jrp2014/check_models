# Automated Review Digest

_Generated on 2026-05-16 23:30:18 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 40 | Keywords 0 | Δ-67 | 30.1 tps | context ignored, degeneration, harness, missing sections
- `mlx-community/Qwen3.5-35B-A3B-6bit`: ❌ F (0/100) | Desc 41 | Keywords 0 | Δ-67 | 86.9 tps | context ignored, cutoff, degeneration, harness, long context, missing sections
- `mlx-community/Qwen3.5-27B-4bit`: ❌ F (0/100) | Desc 40 | Keywords 0 | Δ-67 | 26.8 tps | context ignored, cutoff, degeneration, harness, long context, missing sections
- `mlx-community/Qwen3.5-27B-mxfp8`: ❌ F (0/100) | Desc 40 | Keywords 0 | Δ-67 | 17.8 tps | context ignored, cutoff, degeneration, harness, long context, missing sections
- `mlx-community/InternVL3-14B-8bit`: ❌ F (4/100) | Desc 23 | Keywords 0 | Δ-64 | 37.3 tps | context ignored, harness

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

- None — no models produced clean output meeting all quality thresholds for this prompt.

### `caveat`

| Model                              | Verdict          | Hint Handling                                        | Key Evidence                                                                  |
|------------------------------------|------------------|------------------------------------------------------|-------------------------------------------------------------------------------|
| `mlx-community/MolmoPoint-8B-fp16` | `context_budget` | preserves trusted hints \| nonvisual metadata reused | output/prompt=6.70% \| nontext prompt burden=87% \| nonvisual metadata reused |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                             | Key Evidence                                                                                                                                                                                                                                         |
|---------------------------------------------------------|---------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `facebook/pe-av-large`                                  | `runtime_failure`   | not evaluated                                                                                             | model error \| mlx lm model load model                                                                                                                                                                                                               |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `runtime_failure`   | not evaluated                                                                                             | model error \| mlx model load model                                                                                                                                                                                                                  |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `runtime_failure`   | not evaluated                                                                                             | weight mismatch \| mlx model load weight mismatch                                                                                                                                                                                                    |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `harness`           | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | Output appears truncated to about 6 tokens. \| missing terms: scenic, view, looking, through, open                                                                                                                                                   |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                      | missing sections: keywords \| missing terms: scenic, looking, through, open, wrought \| nonvisual metadata reused                                                                                                                                    |
| `mlx-community/FastVLM-0.5B-bf16`                       | `harness`           | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | Output appears truncated to about 3 tokens. \| missing terms: scenic, view, looking, through, open                                                                                                                                                   |
| `qnguyen3/nanoLLaVA`                                    | `model_shortcoming` | preserves trusted hints                                                                                   | missing sections: keywords \| missing terms: looking, through, open, wrought, iron                                                                                                                                                                   |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open \| degeneration=character_loop: 'ore' repeated                                                                          |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `harness`           | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=72% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `model_shortcoming` | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | nontext prompt burden=72% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                                                   |
| `mlx-community/InternVL3-14B-8bit`                      | `harness`           | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | Output appears truncated to about 6 tokens. \| nontext prompt burden=82% \| missing terms: scenic, view, looking, through, open                                                                                                                      |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open \| repetitive token=phrase: "the learning the learning..."                                                              |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open \| degeneration=character_loop: '-3' repeated                                                                           |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `model_shortcoming` | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | nontext prompt burden=72% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                                                   |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open \| repetitive token=phrase: "neurotransmit outbre neurotran..."                                                         |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `cutoff_degraded`   | preserves trusted hints                                                                                   | hit token cap (500) \| nontext prompt burden=72% \| missing sections: title, description, keywords \| missing terms: down, lined, grand, entrance, style                                                                                             |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=68% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `microsoft/Phi-3.5-vision-instruct`                     | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=68% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open \| repetitive token=phrase: "aesthetic patterns, and the..."                                                            |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=84% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=86% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=91% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=88% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/InternVL3-8B-bf16`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=82% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=94% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=85% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=94% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=77% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                                                         |
| `mlx-community/gemma-4-31b-it-4bit`                     | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open \| degeneration=repeated_punctuation: '\_\_\_\_\_\_...'                                                                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 46 occurrences). \| hit token cap (500) \| nontext prompt burden=84% \| missing sections: title, description, keywords                                                    |
| `mlx-community/pixtral-12b-8bit`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open \| degeneration=repeated_punctuation: '##########...'                                                                   |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `model_shortcoming` | preserves trusted hints                                                                                   | nontext prompt burden=72% \| missing sections: title \| missing terms: scenic, view, looking, down, lined \| reasoning leak                                                                                                                          |
| `mlx-community/X-Reasoner-7B-8bit`                      | `harness`           | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| At long prompt length (16715 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97%                                                       |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open \| formatting=Unknown tags: &lt;footer&gt;                                                                              |
| `mlx-community/pixtral-12b-bf16`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/GLM-4.6V-nvfp4`                          | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| nontext prompt burden=94% \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open                                                                                            |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | At long prompt length (16730 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                                                                |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | At long prompt length (16730 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                                          |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | At long prompt length (16715 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                                                                |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | At long prompt length (16730 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                                                                |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | At long prompt length (16730 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                                          |
| `mlx-community/gemma-4-31b-bf16`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open \| degeneration=incomplete_sentence: ends with 'e'                                                                      |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | At long prompt length (16730 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                                          |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: scenic, view, looking, through, open \| degeneration=repeated_punctuation: '##########...'                                                                   |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | At long prompt length (16730 tokens), output may stop following prompt/image context. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                                          |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `harness`           | ignores trusted hints \| missing terms: scenic, view, looking, through, open                              | Special control token &lt;/think&gt; appeared in generated text. \| At long prompt length (16730 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97%                                                              |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                                                       | Affected Models                                            | Issue Draft                                                                                                                              | Evidence Bundle   | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                                                                           | 1: `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)             | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx`                                                    | Weight/config mismatch during model load              | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                                                       | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx_mlx-model-load-weight-mismatch_001.md)   | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-lm`                                                 | Missing module/import during model load               | Model Error \| phase model_load \| ModuleNotFoundError                                                                                                                                                                  | 1: `facebook/pe-av-large`                                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-lm_mlx-lm-model-load-model_001.md)       | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers              | 46 BPE space markers found in decoded text \| prompt=2,586 \| output/prompt=19.33% \| nontext burden=84% \| stop=max_tokens \| hit token cap (500)                                                                      | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_encoding_001.md)                     | -                 | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;/think&gt; \| prompt_tokens=16730, repetitive output \| prompt=16,730 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) \| 2 model cluster | 2: `mlx-community/Qwen3.6-27B-mxfp8` (+1)                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm_stop-token_001.md)                   | -                 | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~3 \| prompt=491 \| output/prompt=0.61% \| nontext burden=15% \| stop=completed \| 4 model cluster                                                                                                      | 4: `mlx-community/FastVLM-0.5B-bf16` (+3)                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_model-config-mlx-vlm_prompt-template_001.md) | -                 | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | prompt_tokens=16715, repetitive output \| prompt=16,715 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) \| 7 model cluster                                                       | 7: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+6)          | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_007_mlx-vlm-mlx_long-context_001.md)             | -                 | Full and reduced reruns avoid context collapse.           |

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


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 6 tokens.; missing terms:
  scenic, view, looking, through, open
- _Tokens:_ prompt 745 tok; estimated text 419 tok; estimated non-text 326
  tok; generated 6 tok; requested max 500 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: scenic, looking,
  through, open, wrought; nonvisual metadata reused
- _Tokens:_ prompt 487 tok; estimated text 419 tok; estimated non-text 68 tok;
  generated 128 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.; missing terms:
  scenic, view, looking, through, open
- _Tokens:_ prompt 491 tok; estimated text 419 tok; estimated non-text 72 tok;
  generated 3 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: looking, through,
  open, wrought, iron
- _Tokens:_ prompt 487 tok; estimated text 419 tok; estimated non-text 68 tok;
  generated 81 tok; requested max 500 tok; stop reason completed


### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=character_loop: 'ore' repeated
- _Tokens:_ prompt 745 tok; estimated text 419 tok; estimated non-text 326
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Check chat-template and EOS defaults first; the output shape is
  not matching the requested contract.
- _Key signals:_ Output is very short relative to prompt size (0.7%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=72%; missing sections: title, description, keywords; missing terms:
  scenic, view, looking, through, open
- _Tokens:_ prompt 1513 tok; estimated text 419 tok; estimated non-text 1094
  tok; generated 11 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=72%; missing sections: title,
  description, keywords; missing terms: scenic, view, looking, through, open
- _Tokens:_ prompt 1513 tok; estimated text 419 tok; estimated non-text 1094
  tok; generated 15 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 6 tokens.; nontext prompt
  burden=82%; missing terms: scenic, view, looking, through, open
- _Tokens:_ prompt 2270 tok; estimated text 419 tok; estimated non-text 1851
  tok; generated 6 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 3082 tok; estimated text 419 tok; estimated non-text 2663
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open; repetitive
  token=phrase: "the learning the learning..."
- _Tokens:_ prompt 748 tok; estimated text 419 tok; estimated non-text 329
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 1695 tok; estimated text 419 tok; estimated non-text 1276
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 1695 tok; estimated text 419 tok; estimated non-text 1276
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=character_loop: '-3' repeated
- _Tokens:_ prompt 762 tok; estimated text 419 tok; estimated non-text 343
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=72%; missing sections: title,
  description, keywords; missing terms: scenic, view, looking, through, open
- _Tokens:_ prompt 1513 tok; estimated text 419 tok; estimated non-text 1094
  tok; generated 21 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open; repetitive
  token=phrase: "neurotransmit outbre neurotran..."
- _Tokens:_ prompt 596 tok; estimated text 419 tok; estimated non-text 177
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=72%; missing
  sections: title, description, keywords; missing terms: down, lined, grand,
  entrance, style
- _Tokens:_ prompt 1479 tok; estimated text 419 tok; estimated non-text 1060
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=68%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 1305 tok; estimated text 419 tok; estimated non-text 886
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=68%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 1305 tok; estimated text 419 tok; estimated non-text 886
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 3083 tok; estimated text 419 tok; estimated non-text 2664
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open; repetitive
  token=phrase: "aesthetic patterns, and the..."
- _Tokens:_ prompt 756 tok; estimated text 419 tok; estimated non-text 337
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=84%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 2690 tok; estimated text 419 tok; estimated non-text 2271
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 3083 tok; estimated text 419 tok; estimated non-text 2664
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 4585 tok; estimated text 419 tok; estimated non-text 4166
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=88%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 3364 tok; estimated text 419 tok; estimated non-text 2945
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=82%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 2270 tok; estimated text 419 tok; estimated non-text 1851
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=94%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 6540 tok; estimated text 419 tok; estimated non-text 6121
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=85%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 2769 tok; estimated text 419 tok; estimated non-text 2350
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=94%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 6540 tok; estimated text 419 tok; estimated non-text 6121
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=77%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 1808 tok; estimated text 419 tok; estimated non-text 1389
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open
- _Tokens:_ prompt 757 tok; estimated text 419 tok; estimated non-text 338
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=repeated_punctuation: '______...'
- _Tokens:_ prompt 762 tok; estimated text 419 tok; estimated non-text 343
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 46 occurrences).; hit token cap (500); nontext prompt
  burden=84%; missing sections: title, description, keywords
- _Tokens:_ prompt 2586 tok; estimated text 419 tok; estimated non-text 2167
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 3273 tok; estimated text 419 tok; estimated non-text 2854
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=repeated_punctuation: '##########...'
- _Tokens:_ prompt 458 tok; estimated text 419 tok; estimated non-text 39 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 1666 tok; estimated text 419 tok; estimated non-text 1247
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=72%; missing sections: title; missing
  terms: scenic, view, looking, down, lined; reasoning leak
- _Tokens:_ prompt 1479 tok; estimated text 419 tok; estimated non-text 1060
  tok; generated 122 tok; requested max 500 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; At long prompt length (16715 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%
- _Tokens:_ prompt 16715 tok; estimated text 419 tok; estimated non-text 16296
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  formatting=Unknown tags: &lt;footer&gt;
- _Tokens:_ prompt 757 tok; estimated text 419 tok; estimated non-text 338
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 3273 tok; estimated text 419 tok; estimated non-text 2854
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 1666 tok; estimated text 419 tok; estimated non-text 1247
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 87% and the output stays weak under that load.
- _Key signals:_ output/prompt=6.70%; nontext prompt burden=87%; nonvisual
  metadata reused
- _Tokens:_ prompt 3283 tok; estimated text 419 tok; estimated non-text 2864
  tok; generated 220 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=94%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Tokens:_ prompt 6540 tok; estimated text 419 tok; estimated non-text 6121
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=97%; missing sections: title, description, keywords
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16715 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16715 tok; estimated text 419 tok; estimated non-text 16296
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=97%; missing sections: title, description, keywords
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=incomplete_sentence: ends with 'e'
- _Tokens:_ prompt 750 tok; estimated text 419 tok; estimated non-text 331
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=97%; missing sections: title, description, keywords
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=repeated_punctuation: '##########...'
- _Tokens:_ prompt 459 tok; estimated text 419 tok; estimated non-text 40 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=97%; missing sections: title, description, keywords
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; At long prompt length (16730 tokens), output became repetitive.; hit
  token cap (500); nontext prompt burden=97%
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

