# Automated Review Digest

_Generated on 2026-05-04 22:51:32 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🏆 A (87/100) | Desc 83 | Keywords 83 | Δ+24 | 177.3 tps
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🏆 A (86/100) | Desc 84 | Keywords 92 | Δ+23 | 17.3 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (86/100) | Desc 93 | Keywords 86 | Δ+23 | 59.1 tps
- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (85/100) | Desc 93 | Keywords 79 | Δ+22 | 109.8 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (83/100) | Desc 87 | Keywords 84 | Δ+20 | 64.4 tps

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (4/100) | Desc 60 | Keywords 0 | Δ-59 | 30.1 tps | harness, missing sections
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (33/100) | Desc 60 | Keywords 0 | Δ-30 | 191.6 tps | context ignored, cutoff, harness, long context, missing sections, repetitive
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (16/100) | Desc 76 | Keywords 0 | Δ-46 | 63.9 tps | missing sections, trusted hint degraded
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (18/100) | Desc 76 | Keywords 0 | Δ-44 | 5.3 tps | missing sections, trusted hint degraded
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (20/100) | Desc 51 | Keywords 48 | Δ-43 | 31.0 tps | context ignored, missing sections

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict     | Hint Handling                                       | Key Evidence                                                                                                                |
|-----------------------------------------------------|-------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`     | improves trusted hints                              | missing terms: classic, style, during, low, tide                                                                            |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `clean`     | improves trusted hints                              | nontext prompt burden=89% \| missing terms: classic, style, during, exposing, vast                                          |
| `mlx-community/InternVL3-8B-bf16`                   | `clean`     | preserves trusted hints                             | nontext prompt burden=86% \| missing terms: during, receded, exposing, vast, expanse                                        |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`     | improves trusted hints                              | missing terms: style, during, receded, exposing, vast                                                                       |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`     | improves trusted hints                              | nontext prompt burden=89% \| missing terms: receded, exposing, vast, expanse, behind                                        |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`     | improves trusted hints                              | nontext prompt burden=89% \| missing terms: style, during, receded, exposing, vast                                          |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `clean`     | improves trusted hints                              | missing terms: classic, style, wooden, estuary, receded                                                                     |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | `token_cap` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=97% \| missing terms: classic, style, during, receded, exposing \| keywords=20 |
| `mlx-community/Qwen3.6-27B-mxfp8`                   | `token_cap` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| nontext prompt burden=97% \| missing terms: style, estuary, during, tide, receded \| keywords=19     |

### `caveat`

| Model                                      | Verdict          | Hint Handling           | Key Evidence                                                                                                                                            |
|--------------------------------------------|------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/LFM2-VL-1.6B-8bit`          | `clean`          | preserves trusted hints | missing terms: classic, style, sailboat, dark, hull \| keywords=9 \| context echo=64%                                                                   |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `context_budget` | degrades trusted hints  | output/prompt=0.57% \| nontext prompt burden=87% \| missing sections: title, description, keywords \| missing terms: moored, calm, estuary, during, low |
| `mlx-community/gemma-3-27b-it-qat-4bit`    | `clean`          | improves trusted hints  | missing terms: classic, style, wooden, during, receded \| keywords=19                                                                                   |
| `mlx-community/Idefics3-8B-Llama3-bf16`    | `clean`          | preserves trusted hints | nontext prompt burden=87% \| context echo=65% \| formatting=Unknown tags: &lt;end_of_utterance&gt;                                                      |
| `mlx-community/InternVL3-14B-8bit`         | `clean`          | preserves trusted hints | nontext prompt burden=86% \| missing terms: floats, peacefully, waiting, rise, again \| context echo=46%                                                |
| `mlx-community/pixtral-12b-8bit`           | `clean`          | preserves trusted hints | nontext prompt burden=90% \| context echo=58%                                                                                                           |
| `mlx-community/pixtral-12b-bf16`           | `clean`          | preserves trusted hints | nontext prompt burden=90% \| context echo=56%                                                                                                           |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                             | Key Evidence                                                                                                                                                                                                                |
|---------------------------------------------------------|---------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `runtime_failure`   | not evaluated                                                                                             | model error \| mlx model load model                                                                                                                                                                                         |
| `facebook/pe-av-large`                                  | `runtime_failure`   | not evaluated                                                                                             | model error \| mlx vlm model load model                                                                                                                                                                                     |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `runtime_failure`   | not evaluated                                                                                             | model error \| mlx model load model                                                                                                                                                                                         |
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure`   | not evaluated                                                                                             | processor error \| model config processor load processor                                                                                                                                                                    |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | preserves trusted hints                                                                                   | missing sections: keywords \| context echo=98%                                                                                                                                                                              |
| `qnguyen3/nanoLLaVA`                                    | `model_shortcoming` | preserves trusted hints                                                                                   | missing sections: keywords \| missing terms: style, hull, calm, estuary, water                                                                                                                                              |
| `mlx-community/FastVLM-0.5B-bf16`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                      | missing sections: keywords \| context echo=44% \| nonvisual metadata reused                                                                                                                                                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `model_shortcoming` | preserves trusted hints                                                                                   | nontext prompt burden=79% \| missing sections: title, description, keywords \| context echo=100%                                                                                                                            |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `model_shortcoming` | preserves trusted hints                                                                                   | nontext prompt burden=79% \| missing sections: title, description, keywords \| context echo=100%                                                                                                                            |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                      | context echo=66% \| nonvisual metadata reused                                                                                                                                                                               |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `model_shortcoming` | ignores trusted hints \| missing terms: classic, style, sailboat, dark, hull                              | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: classic, style, sailboat, dark, hull                                                                                          |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `cutoff_degraded`   | preserves trusted hints                                                                                   | hit token cap (500) \| missing sections: description, keywords \| missing terms: style, during, receded, exposing, vast \| repetitive token=phrase: "mudflats, flags, boat, water,..."                                      |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: classic, style, sailboat, dark, hull \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: classic, style, sailboat, dark, hull \| nonvisual metadata reused                                                                   |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `model_shortcoming` | degrades trusted hints                                                                                    | missing sections: title, description, keywords \| missing terms: classic, style, dark, hull, wooden                                                                                                                         |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `model_shortcoming` | improves trusted hints                                                                                    | missing sections: title, description, keywords                                                                                                                                                                              |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `model_shortcoming` | preserves trusted hints                                                                                   | missing sections: title, description, keywords \| missing terms: style \| context echo=96%                                                                                                                                  |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `model_shortcoming` | ignores trusted hints \| missing terms: classic, style, sailboat, dark, hull                              | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: classic, style, sailboat, dark, hull                                                                                          |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `model_shortcoming` | improves trusted hints                                                                                    | nontext prompt burden=71% \| missing sections: title \| missing terms: during, receded, vast, expanse, algae \| keywords=45                                                                                                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | preserves trusted hints                                                                                   | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 61 occurrences). \| nontext prompt burden=88% \| missing sections: description, keywords \| missing terms: vast, expanse, adorned, small, floats |
| `microsoft/Phi-3.5-vision-instruct`                     | `cutoff_degraded`   | preserves trusted hints                                                                                   | hit token cap (500) \| nontext prompt burden=67% \| missing terms: boat, adorned, string, small, floats \| keyword duplication=90%                                                                                          |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `cutoff_degraded`   | preserves trusted hints                                                                                   | hit token cap (500) \| nontext prompt burden=67% \| missing terms: boat, adorned, string, small, floats \| keyword duplication=90%                                                                                          |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title, description \| missing terms: boat, floats, peacefully, waiting, rise                                                                          |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: classic, style, sailboat, dark, hull                              | hit token cap (500) \| nontext prompt burden=90% \| missing sections: title, description, keywords \| missing terms: classic, style, sailboat, dark, hull                                                                   |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | preserves trusted hints                                                                                   | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, keywords \| missing terms: style, vast, expanse, floats, waiting                                                                               |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `model_shortcoming` | preserves trusted hints                                                                                   | nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: sailboat, estuary, during, exposing, vast                                                                                     |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: during, receded, exposing, vast, expanse                                                               |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | improves trusted hints                                                                                    | hit token cap (500) \| nontext prompt burden=91% \| missing sections: title, description, keywords \| missing terms: classic, style, receded, vast, expanse                                                                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `model_shortcoming` | preserves trusted hints                                                                                   | nontext prompt burden=70% \| missing sections: title, description, keywords \| missing terms: classic, style, wooden, moored, estuary                                                                                       |
| `mlx-community/X-Reasoner-7B-8bit`                      | `cutoff_degraded`   | improves trusted hints                                                                                    | hit token cap (500) \| nontext prompt burden=97% \| missing terms: style, vast, expanse, peacefully, waiting \| keyword duplication=62%                                                                                     |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `cutoff_degraded`   | ignores trusted hints \| missing terms: classic, style, sailboat, dark, hull                              | hit token cap (500) \| nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: classic, style, sailboat, dark, hull                                                                   |
| `mlx-community/GLM-4.6V-nvfp4`                          | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, description, keywords \| missing terms: vast, expanse, behind, vessel, adorned                                                                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | improves trusted hints                                                                                    | hit token cap (500) \| nontext prompt burden=97% \| missing sections: description, keywords \| missing terms: style, during, receded, exposing, vast                                                                        |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | improves trusted hints                                                                                    | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: style, calm, during, receded, vast                                                                     |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: classic, style, sailboat, dark, hull                              | At long prompt length (16901 tokens), output became repetitive. \| hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords                                                       |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `cutoff_degraded`   | preserves trusted hints                                                                                   | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| reasoning leak                                                                                                        |
| `mlx-community/gemma-4-31b-bf16`                        | `cutoff_degraded`   | improves trusted hints                                                                                    | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: style, dark, hull, during, receded                                                                                                  |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | improves trusted hints                                                                                    | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| missing terms: style, calm, during, receded, exposing                                                                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                       | nontext prompt burden=71% \| missing sections: title \| missing terms: during, receded, exposing, vast, expanse \| keyword duplication=41%                                                                                  |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                      | hit token cap (500) \| nontext prompt burden=97% \| missing sections: title, description, keywords \| nonvisual metadata reused                                                                                             |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](../issues/index.md).

| Target                                         | Problem                                               | Affected Models                                            | Issue Draft                                                                                                    | Evidence Bundle   | Fixed When                                                |
|------------------------------------------------|-------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------|
| `mlx`                                          | Weight/config mismatch during model load              | 2: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (+1)                 | [issue draft](../issues/issue_001_mlx_mlx-model-load-model_001.md)                                             | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | Missing module/import during model load               | 1: `facebook/pe-av-large`                                  | [issue draft](../issues/issue_002_mlx-vlm_mlx-vlm-model-load-model_001.md)                                     | -                 | Load/generation completes or fails with a narrower owner. |
| model configuration / repository               | Processor config is missing image processor           | 1: `mlx-community/MolmoPoint-8B-fp16`                      | [issue draft](../issues/issue_003_model-configuration-repository_model-config-processor-load-processor_001.md) | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | Tokenizer decode leaked BPE/byte markers              | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](../issues/issue_004_mlx-vlm_encoding_001.md)                                                     | -                 | No BPE/byte markers in output.                            |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short | 1: `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [issue draft](../issues/issue_005_mlx-vlm-mlx_long-context_001.md)                                             | -                 | Full and reduced reruns avoid context collapse.           |

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
- _Key signals:_ missing sections: keywords; context echo=98%
- _Tokens:_ prompt 513 tok; estimated text 444 tok; estimated non-text 69 tok;
  generated 90 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: classic, style, sailboat, dark, hull;
  keywords=9; context echo=64%
- _Tokens:_ prompt 767 tok; estimated text 444 tok; estimated non-text 323
  tok; generated 110 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: style, hull, calm,
  estuary, water
- _Tokens:_ prompt 513 tok; estimated text 444 tok; estimated non-text 69 tok;
  generated 64 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; context echo=44%; nonvisual
  metadata reused
- _Tokens:_ prompt 517 tok; estimated text 444 tok; estimated non-text 73 tok;
  generated 181 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=79%; missing sections: title,
  description, keywords; context echo=100%
- _Tokens:_ prompt 2087 tok; estimated text 444 tok; estimated non-text 1643
  tok; generated 70 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=79%; missing sections: title,
  description, keywords; context echo=100%
- _Tokens:_ prompt 2087 tok; estimated text 444 tok; estimated non-text 1643
  tok; generated 70 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: classic, style, during, low, tide
- _Tokens:_ prompt 784 tok; estimated text 444 tok; estimated non-text 340
  tok; generated 89 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ context echo=66%; nonvisual metadata reused
- _Tokens:_ prompt 623 tok; estimated text 444 tok; estimated non-text 179
  tok; generated 137 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=71%; missing sections: title,
  description, keywords; missing terms: classic, style, sailboat, dark, hull
- _Tokens:_ prompt 1538 tok; estimated text 444 tok; estimated non-text 1094
  tok; generated 31 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=89%; missing terms: classic, style,
  during, exposing, vast
- _Tokens:_ prompt 4114 tok; estimated text 444 tok; estimated non-text 3670
  tok; generated 127 tok; requested max 500 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 87% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.57%; nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: moored, calm,
  estuary, during, low
- _Tokens:_ prompt 3503 tok; estimated text 444 tok; estimated non-text 3059
  tok; generated 20 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: description, keywords;
  missing terms: style, during, receded, exposing, vast; repetitive
  token=phrase: "mudflats, flags, boat, water,..."
- _Tokens:_ prompt 575 tok; estimated text 444 tok; estimated non-text 131
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=86%; missing terms: during, receded,
  exposing, vast, expanse
- _Tokens:_ prompt 3064 tok; estimated text 444 tok; estimated non-text 2620
  tok; generated 87 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: classic, style, sailboat, dark, hull; nonvisual
  metadata reused
- _Tokens:_ prompt 774 tok; estimated text 444 tok; estimated non-text 330
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: style, during, receded, exposing, vast
- _Tokens:_ prompt 784 tok; estimated text 444 tok; estimated non-text 340
  tok; generated 88 tok; requested max 500 tok; stop reason completed


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: classic, style, dark, hull, wooden
- _Tokens:_ prompt 484 tok; estimated text 444 tok; estimated non-text 40 tok;
  generated 17 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=89%; missing terms: receded, exposing,
  vast, expanse, behind
- _Tokens:_ prompt 4115 tok; estimated text 444 tok; estimated non-text 3671
  tok; generated 114 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: classic, style, wooden, during, receded;
  keywords=19
- _Tokens:_ prompt 783 tok; estimated text 444 tok; estimated non-text 339
  tok; generated 113 tok; requested max 500 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=87%; context echo=65%;
  formatting=Unknown tags: &lt;end_of_utterance&gt;
- _Tokens:_ prompt 3507 tok; estimated text 444 tok; estimated non-text 3063
  tok; generated 117 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=89%; missing terms: style, during,
  receded, exposing, vast
- _Tokens:_ prompt 4115 tok; estimated text 444 tok; estimated non-text 3671
  tok; generated 118 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords
- _Tokens:_ prompt 782 tok; estimated text 444 tok; estimated non-text 338
  tok; generated 266 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=86%; missing terms: floats, peacefully,
  waiting, rise, again; context echo=46%
- _Tokens:_ prompt 3064 tok; estimated text 444 tok; estimated non-text 2620
  tok; generated 120 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=90%; context echo=58%
- _Tokens:_ prompt 4662 tok; estimated text 444 tok; estimated non-text 4218
  tok; generated 135 tok; requested max 500 tok; stop reason completed


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: style; context echo=96%
- _Tokens:_ prompt 483 tok; estimated text 444 tok; estimated non-text 39 tok;
  generated 119 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=71%; missing sections: title,
  description, keywords; missing terms: classic, style, sailboat, dark, hull
- _Tokens:_ prompt 1538 tok; estimated text 444 tok; estimated non-text 1094
  tok; generated 31 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: classic, style, wooden, estuary, receded
- _Tokens:_ prompt 783 tok; estimated text 444 tok; estimated non-text 339
  tok; generated 104 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=71%; missing sections: title; missing
  terms: during, receded, vast, expanse, algae; keywords=45
- _Tokens:_ prompt 1516 tok; estimated text 444 tok; estimated non-text 1072
  tok; generated 490 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=90%; context echo=56%
- _Tokens:_ prompt 4662 tok; estimated text 444 tok; estimated non-text 4218
  tok; generated 138 tok; requested max 500 tok; stop reason completed


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 61 occurrences).; nontext prompt burden=88%; missing sections:
  description, keywords; missing terms: vast, expanse, adorned, small, floats
- _Tokens:_ prompt 3619 tok; estimated text 444 tok; estimated non-text 3175
  tok; generated 108 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=67%; missing
  terms: boat, adorned, string, small, floats; keyword duplication=90%
- _Tokens:_ prompt 1337 tok; estimated text 444 tok; estimated non-text 893
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=67%; missing
  terms: boat, adorned, string, small, floats; keyword duplication=90%
- _Tokens:_ prompt 1337 tok; estimated text 444 tok; estimated non-text 893
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description; missing terms: boat, floats, peacefully,
  waiting, rise
- _Tokens:_ prompt 1810 tok; estimated text 444 tok; estimated non-text 1366
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: classic, style,
  sailboat, dark, hull
- _Tokens:_ prompt 4610 tok; estimated text 444 tok; estimated non-text 4166
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, keywords; missing terms: style, vast, expanse, floats,
  waiting
- _Tokens:_ prompt 6570 tok; estimated text 444 tok; estimated non-text 6126
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title,
  description, keywords; missing terms: sailboat, estuary, during, exposing,
  vast
- _Tokens:_ prompt 1468 tok; estimated text 444 tok; estimated non-text 1024
  tok; generated 190 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: during, receded,
  exposing, vast, expanse
- _Tokens:_ prompt 6570 tok; estimated text 444 tok; estimated non-text 6126
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: classic, style,
  receded, vast, expanse
- _Tokens:_ prompt 4753 tok; estimated text 444 tok; estimated non-text 4309
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title,
  description, keywords; missing terms: classic, style, wooden, moored,
  estuary
- _Tokens:_ prompt 1468 tok; estimated text 444 tok; estimated non-text 1024
  tok; generated 163 tok; requested max 500 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: style, vast, expanse, peacefully, waiting; keyword duplication=62%
- _Tokens:_ prompt 16901 tok; estimated text 444 tok; estimated non-text 16457
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=71%; missing
  sections: title, description, keywords; missing terms: classic, style,
  sailboat, dark, hull
- _Tokens:_ prompt 1538 tok; estimated text 444 tok; estimated non-text 1094
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: vast, expanse,
  behind, vessel, adorned
- _Tokens:_ prompt 6570 tok; estimated text 444 tok; estimated non-text 6126
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: description, keywords; missing terms: style, during, receded,
  exposing, vast
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: style, calm, during,
  receded, vast
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16901 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 16901 tok; estimated text 444 tok; estimated non-text 16457
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; reasoning leak
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: style, dark, hull, during, receded
- _Tokens:_ prompt 772 tok; estimated text 444 tok; estimated non-text 328
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: style, calm, during,
  receded, exposing
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=71%; missing sections: title; missing
  terms: during, receded, exposing, vast, expanse; keyword duplication=41%
- _Tokens:_ prompt 1516 tok; estimated text 444 tok; estimated non-text 1072
  tok; generated 351 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; nonvisual metadata reused
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: classic, style, during, receded, exposing; keywords=20
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: style, estuary, during, tide, receded; keywords=19
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed

