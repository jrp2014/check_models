<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

Generated on: 2026-07-19 20:37:23 BST

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Companion artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Qwen3.5-35B-A3B-bf16`: 🏆 A (90/100) | Desc 78 | Keywords 74 | Δ+16 | 65.8 tps
- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (88/100) | Desc 92 | Keywords 58 | Δ+14 | 123.2 tps
- `mlx-community/Qwen3.5-35B-A3B-6bit`: 🏆 A (82/100) | Desc 78 | Keywords 74 | Δ+8 | 91.2 tps
- `mlx-community/Qwen3.5-27B-4bit`: 🏆 A (81/100) | Desc 99 | Keywords 75 | Δ+7 | 30.7 tps
- `mlx-community/Qwen3.5-9B-MLX-4bit`: ✅ B (77/100) | Desc 77 | Keywords 59 | Δ+2 | 92.1 tps

### Watchlist

- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-74 | 103.0 tps | context ignored, harness, long context, low metadata alignment
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (6/100) | Desc 45 | Keywords 0 | Δ-68 | 64.1 tps | context ignored, harness, long context, low metadata alignment
- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-54 | 93.3 tps | context ignored, cutoff, degeneration, generation loop, harness, long context, low metadata alignment, missing sections, repetitive, text sanity
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-54 | 92.3 tps | context ignored, cutoff, degeneration, generation loop, harness, long context, low metadata alignment, missing sections, repetitive, text sanity
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (45/100) | Desc 60 | Keywords 42 | Δ-29 | 225.1 tps | context ignored, cutoff, generation loop, harness, long context, low metadata alignment, missing sections, repetitive, unverified context copy

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict   | Hint Handling                                    | Key Evidence                                                                                                                           |
|-----------------------------------------------------|-----------|--------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`   | preserves trusted hints                          | missing terms: Cars, City, Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie)                                         |
| `mlx-community/InternVL3-8B-bf16`                   | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Commuting, Fenchurch Street, Nightscape, The Fenchurch Building (The Walkie-Talkie), Fenchurch \| low-draft-improvement |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`   | preserves trusted hints                          | missing terms: Cars, City, Commuting, Nightscape, Street signs                                                                         |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Cars, Commuting, The Fenchurch Building (The Walkie-Talkie), GBR, formally \| low-draft-improvement                     |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Cars, Commuting, Street signs, GBR, formally \| low-draft-improvement                                                   |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                | `clean`   | preserves trusted hints                          | missing terms: Commuting, Fenchurch Street, Illuminated, GBR, known                                                                    |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`   | preserves trusted hints                          | missing terms: Cars, City, Commuting, Fenchurch Street, Nightscape                                                                     |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `clean`   | preserves trusted hints                          | missing terms: Cars, Commuting, The Fenchurch Building (The Walkie-Talkie), GBR, formally                                              |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                | `clean`   | preserves trusted hints                          | missing terms: Cars, City, Commuting, Fenchurch Street, Nightscape                                                                     |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `clean`   | preserves trusted hints                          | missing terms: Cars, Cityscape, Commuting, Modern, The Fenchurch Building (The Walkie-Talkie)                                          |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `clean`   | preserves trusted hints                          | missing terms: Cars, City, Commuting, Modern, The Fenchurch Building (The Walkie-Talkie)                                               |
| `mlx-community/Qwen3.5-27B-4bit`                    | `clean`   | preserves trusted hints                          | missing terms: Building, Buildings, Cars, City, Commuting                                                                              |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | `clean`   | preserves trusted hints                          | missing terms: Cars, City, Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie)                                         |
| `mlx-community/Qwen3.6-27B-mxfp8`                   | `clean`   | preserves trusted hints                          | missing terms: Cars, City, Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie)                                         |
| `mlx-community/Ornith-1.0-35B-bf16`                 | `clean`   | preserves trusted hints                          | missing terms: Building, Buildings, Commuting, Modern, Nightscape                                                                      |

### `caveat`

| Model                                                   | Verdict             | Hint Handling                                                                                                  | Key Evidence                                                                                                                                                                                                |
|---------------------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | missing terms: Cars, Commuting \| keywords=23 \| low-draft-improvement                                                                                                                                      |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | keywords=20 \| context echo=76% \| low-draft-improvement                                                                                                                                                    |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | missing terms: formally \| keywords=20 \| low-draft-improvement                                                                                                                                             |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | keywords=20 \| low-draft-improvement                                                                                                                                                                        |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | `clean`             | preserves trusted hints                                                                                        | missing terms: Commuting, Nightscape, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban landscape \| keywords=22                                                                              |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          | `model_shortcoming` | preserves trusted hints                                                                                        | missing sections: title \| missing terms: Cars, City, Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie) \| formatting=Unknown tags: &lt;channel\|&gt;                                     |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `model_shortcoming` | degrades trusted hints \| low-draft-improvement                                                                | missing sections: title, description, keywords \| missing terms: Architecture, Cars, City, Cityscape, Commuting \| formatting=Unknown tags: &lt;end_of_utterance&gt; \| low-draft-improvement               |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         | `model_shortcoming` | preserves trusted hints \| low-draft-improvement                                                               | missing sections: title \| missing terms: Cars, Cityscape, Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie) \| formatting=Unknown tags: &lt;channel\|&gt; \| low-draft-improvement       |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | missing terms: known, formally, 20 \| keywords=20 \| low-draft-improvement                                                                                                                                  |
| `mlx-community/pixtral-12b-8bit`                        | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | missing terms: Commuting, The Fenchurch Building (The Walkie-Talkie), GBR, formally \| keywords=19 \| low-draft-improvement                                                                                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | missing terms: Cars, Commuting, GBR \| keywords=23 \| low-draft-improvement                                                                                                                                 |
| `mlx-community/pixtral-12b-bf16`                        | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | missing terms: Cars, Commuting, Modern, The Fenchurch Building (The Walkie-Talkie), Walkie Talkie building \| low-draft-improvement                                                                         |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | `clean`             | preserves trusted hints                                                                                        | missing terms: Cars, Cityscape, Commuting, Fenchurch Street, Nightscape \| formatting=Empty thinking wrapper present                                                                                        |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | missing terms: Commuting, GBR, formally, 20 \| keywords=19 \| low-draft-improvement                                                                                                                         |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `context_budget`    | preserves trusted hints \| low-draft-improvement                                                               | output/prompt=2.29% \| visual input burden=92% \| missing sections: title \| missing terms: Commuting, The Fenchurch Building (The Walkie-Talkie), GBR, known, formally                                     |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | missing terms: Cars, Commuting, Walkie Talkie building, GBR, formally \| keywords=21 \| low-draft-improvement                                                                                               |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | `context_budget`    | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | Output appears truncated to about 9 tokens. \| At mixed burden (16844 tokens), output stayed unusually short (9 tokens; ratio 0.1%; weak text signal truncated). \| output/prompt=0.05% \| mixed burden=97% |
| `mlx-community/GLM-4.6V-nvfp4`                          | `context_budget`    | preserves trusted hints \| low-draft-improvement                                                               | output/prompt=2.20% \| visual input burden=92% \| missing sections: title \| missing terms: Commuting, known, formally, 20                                                                                  |
| `mlx-community/X-Reasoner-7B-8bit`                      | `context_budget`    | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | Output appears truncated to about 9 tokens. \| At mixed burden (16853 tokens), output stayed unusually short (9 tokens; ratio 0.1%; weak text signal truncated). \| output/prompt=0.05% \| mixed burden=97% |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | missing terms: formally, 20 \| keywords=20 \| low-draft-improvement                                                                                                                                         |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | keywords=20 \| low-draft-improvement                                                                                                                                                                        |
| `mlx-community/MolmoPoint-8B-fp16`                      | `clean`             | preserves trusted hints \| low-draft-improvement                                                               | missing terms: formally \| keywords=20 \| low-draft-improvement                                                                                                                                             |

### `needs_triage`

- None.

### `avoid`

| Model                                              | Verdict             | Hint Handling                                                                                                  | Key Evidence                                                                                                                                                                                       |
|----------------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Step-3.7-Flash-oQ2e`                | `runtime_failure`   | not evaluated                                                                                                  | processor error \| model config processor load processor                                                                                                                                           |
| `HuggingFaceTB/SmolVLM-Instruct`                   | `model_shortcoming` | degrades trusted hints \| low-draft-improvement                                                                | missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement                                                            |
| `mlx-community/SmolVLM-Instruct-bf16`              | `model_shortcoming` | degrades trusted hints \| low-draft-improvement                                                                | missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement                                                            |
| `qnguyen3/nanoLLaVA`                               | `model_shortcoming` | degrades trusted hints \| low-draft-improvement                                                                | missing sections: keywords \| missing terms: Architecture, Cars, City, Cityscape, Commuting \| low-draft-improvement                                                                               |
| `mlx-community/FastVLM-0.5B-bf16`                  | `model_shortcoming` | preserves trusted hints \| low-draft-improvement                                                               | missing sections: keywords \| missing terms: Cars, Cityscape, Commuting, Modern, Nightscape \| low-draft-improvement                                                                               |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`   | `model_shortcoming` | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement                                                            |
| `mlx-community/gemma-3n-E2B-4bit`                  | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City \| repetitive token=phrase: "- do not take..."               |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`         | `model_shortcoming` | degrades trusted hints \| low-draft-improvement                                                                | missing sections: title, description, keywords \| missing terms: Architecture, Cars, City, Cityscape, Commuting \| low-draft-improvement                                                           |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`   | `model_shortcoming` | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement                                                            |
| `mlx-community/gemma-3n-E4B-it-bf16`               | `model_shortcoming` | preserves trusted hints \| unverified-context-copy                                                             | missing terms: Commuting, Fenchurch Street, Nightscape, Street signs, The Fenchurch Building (The Walkie-Talkie) \| keywords=25 \| unverified-context-copy                                         |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`            | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                               | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: City, Commuting, Skyscraper, Urban landscape, known \| thinking trace incomplete                           |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`          | `cutoff_degraded`   | preserves trusted hints                                                                                        | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Cars, City, Nightscape, Street signs, Urban \| thinking trace present                                      |
| `microsoft/Phi-3.5-vision-instruct`                | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                               | hit token cap (500) \| missing terms: Cars, Commuting, Fenchurch Street, Nightscape, Street \| keyword duplication=83% \| repetitive token=phrase: "modern, illuminated, architect..."             |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                               | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: City, Commuting, GBR, known, formally \| text-sanity=numeric_loop                                          |
| `mlx-community/paligemma2-3b-pt-896-4bit`          | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City \| excessive bullets=42                                      |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`    | `cutoff_degraded`   | preserves trusted hints                                                                                        | hit token cap (500) \| missing sections: title, keywords \| missing terms: Architecture, Cars, City, Commuting, Modern \| reasoning leak                                                           |
| `mlx-community/gemma-4-31b-bf16`                   | `model_shortcoming` | degrades trusted hints \| low-draft-improvement                                                                | missing sections: title, description, keywords \| missing terms: Architecture, Cityscape, Commuting, Fenchurch Street, London \| low-draft-improvement                                             |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                | `model_shortcoming` | preserves trusted hints \| unverified-context-copy \| low-draft-improvement                                    | missing terms: formally \| keywords=30 \| unverified-context-copy \| low-draft-improvement                                                                                                         |
| `Qwen/Qwen3-VL-2B-Instruct`                        | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | At mixed burden (16842 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`    | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement                                     |
| `mlx-community/Phi-3.5-vision-instruct-bf16`       | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                               | hit token cap (500) \| missing terms: Cars, Commuting, Fenchurch Street, Nightscape, Street \| keyword duplication=83% \| repetitive token=phrase: "modern, illuminated, architect..."             |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`          | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | At mixed burden (16842 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`     | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                               | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: known, formally, 20 \| thinking trace incomplete                                                           |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`          | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | At mixed burden (16853 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

<!-- markdownlint-disable MD060 -->

| Target                           | Problem                                     | Evidence Snapshot                                                                                                                                  | Affected Models                                       | Issue Draft                                                                                                                                                              | Evidence Bundle                                                                                                                                                                               | Fixed When                                                   |
|----------------------------------|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| model configuration / repository | Processor config is missing image processor | Processor Error \| phase processor_load \| ValueError                                                                                              | 1: `mlx-community/Step-3.7-Flash-oQ2e`                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_model-configuration-repository_model-config-processor-load-processor_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260719T193722Z_002_mlx-community_Step-3.7-Flash-oQ2e_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_3c.json) | Load/generation completes or fails with a narrower owner.    |
| model repository                 | Generated text is mixed-script token-soup   | token cap \| missing sections \| abrupt tail \| prompt=3,524 \| output/prompt=14.19% \| mixed burden=85% \| stop=max_tokens \| hit token cap (500) | 1: `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`    | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_model_text-sanity_001.md)                                                    | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260719T193722Z_004_mlx-community_Apriel-1.5-15b-Thinker-6bit-MLX_model_text_sanity_001.json)        | Generated text is readable natural language, not token soup. |
| model repository                 | Generated text is mixed-script token-soup   | token cap \| missing sections \| abrupt tail \| prompt=1,956 \| output/prompt=25.56% \| stop=max_tokens \| hit token cap (500)                     | 1: `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_model_text-sanity_002.md)                                                    | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260719T193722Z_003_mlx-community_ERNIE-4.5-VL-28B-A3B-Thinking-bf16_model_text_sanity_002.json)     | Generated text is readable natural language, not token soup. |
<!-- markdownlint-enable MD060 -->

## Model Verdicts

### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ not evaluated; review verdict: indeterminate
- _Owner:_ likely owner `unknown`; reported package `unknown`; failure stage
  `Network Error`; diagnostic code `UNKNOWN_MODEL_LOAD_NETWORK_ERROR`
- _Next step:_ Retry when external connectivity is stable; the model was not
  evaluated and the disconnect does not identify a faulty package.
- _Key signals:_ network error; unknown model load network error; external
  connectivity
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `mlx-community/Step-3.7-Flash-oQ2e`

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


### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting; keywords=23;
  low-draft-improvement
- _Tokens:_ prompt 700 tok; estimated text 517 tok; estimated non-text 183
  tok; generated 188 tok; requested max 500 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; context echo=76%; low-draft-improvement
- _Tokens:_ prompt 625 tok; estimated text 517 tok; estimated non-text 108
  tok; generated 112 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: formally; keywords=20; low-draft-improvement
- _Tokens:_ prompt 890 tok; estimated text 517 tok; estimated non-text 373
  tok; generated 201 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City; low-draft-improvement
- _Tokens:_ prompt 1843 tok; estimated text 517 tok; estimated non-text 1326
  tok; generated 18 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City; low-draft-improvement
- _Tokens:_ prompt 1843 tok; estimated text 517 tok; estimated non-text 1326
  tok; generated 18 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Architecture,
  Cars, City, Cityscape, Commuting; low-draft-improvement
- _Tokens:_ prompt 625 tok; estimated text 517 tok; estimated non-text 108
  tok; generated 91 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Cars, Cityscape,
  Commuting, Modern, Nightscape; low-draft-improvement
- _Tokens:_ prompt 629 tok; estimated text 517 tok; estimated non-text 112
  tok; generated 179 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Street signs, The
  Fenchurch Building (The Walkie-Talkie)
- _Tokens:_ prompt 905 tok; estimated text 517 tok; estimated non-text 388
  tok; generated 101 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; low-draft-improvement
- _Tokens:_ prompt 743 tok; estimated text 517 tok; estimated non-text 226
  tok; generated 158 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City; low-draft-improvement
- _Tokens:_ prompt 1652 tok; estimated text 517 tok; estimated non-text 1135
  tok; generated 17 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, Nightscape, Street signs, The
  Fenchurch Building (The Walkie-Talkie), Urban landscape; keywords=22
- _Tokens:_ prompt 3239 tok; estimated text 517 tok; estimated non-text 2722
  tok; generated 151 tok; requested max 500 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-8bit`

- _Recommendation:_ use with caveats; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Cars, City,
  Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie);
  formatting=Unknown tags: &lt;channel|&gt;
- _Tokens:_ prompt 901 tok; estimated text 517 tok; estimated non-text 384
  tok; generated 104 tok; requested max 500 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ use with caveats; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Cars, City, Cityscape, Commuting; formatting=Unknown
  tags: &lt;end_of_utterance&gt;; low-draft-improvement
- _Tokens:_ prompt 2907 tok; estimated text 517 tok; estimated non-text 2390
  tok; generated 26 tok; requested max 500 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

- _Recommendation:_ use with caveats; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Cars, Cityscape,
  Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie);
  formatting=Unknown tags: &lt;channel|&gt;; low-draft-improvement
- _Tokens:_ prompt 901 tok; estimated text 517 tok; estimated non-text 384
  tok; generated 100 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, Fenchurch Street, Nightscape, The
  Fenchurch Building (The Walkie-Talkie), Fenchurch; low-draft-improvement
- _Tokens:_ prompt 2408 tok; estimated text 517 tok; estimated non-text 1891
  tok; generated 90 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Building, Buildings, Cars, City;
  repetitive token=phrase: "- do not take..."
- _Tokens:_ prompt 891 tok; estimated text 517 tok; estimated non-text 374
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Cars, City, Cityscape, Commuting; low-draft-improvement
- _Tokens:_ prompt 596 tok; estimated text 517 tok; estimated non-text 79 tok;
  generated 14 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City; low-draft-improvement
- _Tokens:_ prompt 1652 tok; estimated text 517 tok; estimated non-text 1135
  tok; generated 17 tok; requested max 500 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: known, formally, 20; keywords=20;
  low-draft-improvement
- _Tokens:_ prompt 2867 tok; estimated text 517 tok; estimated non-text 2350
  tok; generated 161 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Nightscape, Street
  signs
- _Tokens:_ prompt 3240 tok; estimated text 517 tok; estimated non-text 2723
  tok; generated 158 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, The Fenchurch Building (The
  Walkie-Talkie), GBR, formally; keywords=19; low-draft-improvement
- _Tokens:_ prompt 3433 tok; estimated text 517 tok; estimated non-text 2916
  tok; generated 112 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, The Fenchurch Building (The
  Walkie-Talkie), GBR, formally; low-draft-improvement
- _Tokens:_ prompt 2408 tok; estimated text 517 tok; estimated non-text 1891
  tok; generated 118 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, GBR; keywords=23;
  low-draft-improvement
- _Tokens:_ prompt 900 tok; estimated text 517 tok; estimated non-text 383
  tok; generated 142 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Street signs, GBR, formally;
  low-draft-improvement
- _Tokens:_ prompt 905 tok; estimated text 517 tok; estimated non-text 388
  tok; generated 128 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Modern, The Fenchurch
  Building (The Walkie-Talkie), Walkie Talkie building; low-draft-improvement
- _Tokens:_ prompt 3433 tok; estimated text 517 tok; estimated non-text 2916
  tok; generated 99 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, Fenchurch Street, Nightscape,
  Street signs, The Fenchurch Building (The Walkie-Talkie); keywords=25;
  unverified-context-copy
- _Tokens:_ prompt 899 tok; estimated text 517 tok; estimated non-text 382
  tok; generated 323 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, Fenchurch Street, Illuminated, GBR,
  known
- _Tokens:_ prompt 6647 tok; estimated text 517 tok; estimated non-text 6130
  tok; generated 134 tok; requested max 500 tok; stop reason completed


### `mlx-community/MiniCPM-V-4.6-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Cityscape, Commuting, Fenchurch Street,
  Nightscape; formatting=Empty thinking wrapper present
- _Tokens:_ prompt 1243 tok; estimated text 517 tok; estimated non-text 726
  tok; generated 94 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: City, Commuting, Skyscraper, Urban landscape,
  known; thinking trace incomplete
- _Tokens:_ prompt 1623 tok; estimated text 517 tok; estimated non-text 1106
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, GBR, formally, 20; keywords=19;
  low-draft-improvement
- _Tokens:_ prompt 2707 tok; estimated text 517 tok; estimated non-text 2190
  tok; generated 136 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Run a controlled reduced-image or lower-visual-token comparison
  before assigning the context-boundary behaviour to mlx, mlx-vlm, or the
  model.
- _Key signals:_ output/prompt=2.29%; visual input burden=92%; missing
  sections: title; missing terms: Commuting, The Fenchurch Building (The
  Walkie-Talkie), GBR, known, formally
- _Tokens:_ prompt 6647 tok; estimated text 517 tok; estimated non-text 6130
  tok; generated 152 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Cars, City, Nightscape, Street signs, Urban;
  thinking trace present
- _Tokens:_ prompt 1623 tok; estimated text 517 tok; estimated non-text 1106
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: Cars, Commuting,
  Fenchurch Street, Nightscape, Street; keyword duplication=83%; repetitive
  token=phrase: "modern, illuminated, architect..."
- _Tokens:_ prompt 1468 tok; estimated text 517 tok; estimated non-text 951
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Walkie Talkie building, GBR,
  formally; keywords=21; low-draft-improvement
- _Tokens:_ prompt 900 tok; estimated text 517 tok; estimated non-text 383
  tok; generated 140 tok; requested max 500 tok; stop reason completed


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: City, Commuting, GBR, known, formally;
  text-sanity=numeric_loop
- _Tokens:_ prompt 1956 tok; estimated text 517 tok; estimated non-text 1439
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Building, Buildings, Cars, City;
  excessive bullets=42
- _Tokens:_ prompt 4724 tok; estimated text 517 tok; estimated non-text 4207
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Run a controlled reduced-image or lower-visual-token comparison
  before assigning the context-boundary behaviour to mlx, mlx-vlm, or the
  model.
- _Key signals:_ Output appears truncated to about 9 tokens.; At mixed burden
  (16844 tokens), output stayed unusually short (9 tokens; ratio 0.1%; weak
  text signal truncated).; output/prompt=0.05%; mixed burden=97%
- _Tokens:_ prompt 16844 tok; estimated text 517 tok; estimated non-text 16327
  tok; generated 9 tok; requested max 500 tok; stop reason completed


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, keywords;
  missing terms: Architecture, Cars, City, Commuting, Modern; reasoning leak
- _Tokens:_ prompt 3524 tok; estimated text 517 tok; estimated non-text 3007
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Run a controlled reduced-image or lower-visual-token comparison
  before assigning the context-boundary behaviour to mlx, mlx-vlm, or the
  model.
- _Key signals:_ output/prompt=2.20%; visual input burden=92%; missing
  sections: title; missing terms: Commuting, known, formally, 20
- _Tokens:_ prompt 6647 tok; estimated text 517 tok; estimated non-text 6130
  tok; generated 146 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Cityscape, Commuting, Fenchurch Street, London;
  low-draft-improvement
- _Tokens:_ prompt 893 tok; estimated text 517 tok; estimated non-text 376
  tok; generated 82 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: formally; keywords=30;
  unverified-context-copy; low-draft-improvement
- _Tokens:_ prompt 890 tok; estimated text 517 tok; estimated non-text 373
  tok; generated 294 tok; requested max 500 tok; stop reason completed


### `Qwen/Qwen3-VL-2B-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16842 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City
- _Tokens:_ prompt 16842 tok; estimated text 517 tok; estimated non-text 16325
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Run a controlled reduced-image or lower-visual-token comparison
  before assigning the context-boundary behaviour to mlx, mlx-vlm, or the
  model.
- _Key signals:_ Output appears truncated to about 9 tokens.; At mixed burden
  (16853 tokens), output stayed unusually short (9 tokens; ratio 0.1%; weak
  text signal truncated).; output/prompt=0.05%; mixed burden=97%
- _Tokens:_ prompt 16853 tok; estimated text 517 tok; estimated non-text 16336
  tok; generated 9 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Fenchurch Street,
  Nightscape
- _Tokens:_ prompt 3240 tok; estimated text 517 tok; estimated non-text 2723
  tok; generated 122 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Building, Buildings, Cars, City;
  low-draft-improvement
- _Tokens:_ prompt 1652 tok; estimated text 517 tok; estimated non-text 1135
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: formally, 20; keywords=20;
  low-draft-improvement
- _Tokens:_ prompt 595 tok; estimated text 517 tok; estimated non-text 78 tok;
  generated 187 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: Cars, Commuting,
  Fenchurch Street, Nightscape, Street; keyword duplication=83%; repetitive
  token=phrase: "modern, illuminated, architect..."
- _Tokens:_ prompt 1468 tok; estimated text 517 tok; estimated non-text 951
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; low-draft-improvement
- _Tokens:_ prompt 1812 tok; estimated text 517 tok; estimated non-text 1295
  tok; generated 175 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, The Fenchurch Building (The
  Walkie-Talkie), GBR, formally
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 137 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16842 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City
- _Tokens:_ prompt 16842 tok; estimated text 517 tok; estimated non-text 16325
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Fenchurch Street,
  Nightscape
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 133 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Cityscape, Commuting, Modern, The
  Fenchurch Building (The Walkie-Talkie)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 132 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Modern, The Fenchurch
  Building (The Walkie-Talkie)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 131 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Building, Buildings, Cars, City, Commuting
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 136 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Street signs, The
  Fenchurch Building (The Walkie-Talkie)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 141 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Street signs, The
  Fenchurch Building (The Walkie-Talkie)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 144 tok; requested max 500 tok; stop reason completed


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: formally; keywords=20; low-draft-improvement
- _Tokens:_ prompt 3440 tok; estimated text 517 tok; estimated non-text 2923
  tok; generated 155 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: known, formally, 20; thinking trace incomplete
- _Tokens:_ prompt 1623 tok; estimated text 517 tok; estimated non-text 1106
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Ornith-1.0-35B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Building, Buildings, Commuting, Modern,
  Nightscape
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 126 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16853 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City
- _Tokens:_ prompt 16853 tok; estimated text 517 tok; estimated non-text 16336
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

