<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

Generated on: 2026-07-19 02:41:57 BST

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Companion artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Ornith-1.0-35B-bf16`: 🏆 A (92/100) | Desc 96 | Keywords 75 | Δ+18 | 63.3 tps
- `mlx-community/Qwen3.5-35B-A3B-bf16`: 🏆 A (92/100) | Desc 79 | Keywords 75 | Δ+18 | 65.0 tps
- `mlx-community/Qwen3.5-27B-4bit`: 🏆 A (92/100) | Desc 90 | Keywords 75 | Δ+17 | 30.7 tps
- `mlx-community/Qwen3.5-35B-A3B-4bit`: 🏆 A (91/100) | Desc 78 | Keywords 75 | Δ+17 | 90.1 tps
- `mlx-community/Qwen3.5-27B-mxfp8`: 🏆 A (87/100) | Desc 93 | Keywords 59 | Δ+13 | 17.4 tps

### Watchlist

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-74 | 49588.4 tps | harness, long context
- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-54 | 92.9 tps | context ignored, cutoff, degeneration, generation loop, harness, long context, missing sections, repetitive, text sanity
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-54 | 92.8 tps | context ignored, cutoff, degeneration, generation loop, harness, long context, missing sections, repetitive, text sanity
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-54 | 92.7 tps | context ignored, cutoff, degeneration, generation loop, harness, long context, missing sections, repetitive, text sanity
- `mlx-community/X-Reasoner-7B-8bit`: 🟠 D (50/100) | Desc 40 | Keywords 0 | Δ-24 | 56.1 tps | context ignored, cutoff, generation loop, hallucination, harness, long context, missing sections, repetitive, text sanity

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                                   | Verdict   | Hint Handling                                    | Key Evidence                                                                                                                  |
|---------------------------------------------------------|-----------|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/InternVL3-8B-bf16`                       | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Nightscape, The Fenchurch Building (The Walki..., GBR, known, formally \| low-draft-improvement                |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Cars, Commuting, Fenchurch Street, Street signs, The Fenchurch Building (The Walki... \| low-draft-improvement |
| `mlx-community/gemma-4-31b-it-4bit`                     | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Cars, Commuting, Street signs, The Fenchurch Building (The Walki..., GBR \| low-draft-improvement              |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Cars, Commuting, Fenchurch Street, Street signs, The Fenchurch Building (The Walki... \| low-draft-improvement |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Cityscape, Commuting, London, Nightscape, The Fenchurch Building (The Walki... \| low-draft-improvement        |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `clean`   | preserves trusted hints                          | missing terms: Cars, Commuting, Fenchurch Street, The Fenchurch Building (The Walki..., GBR                                   |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `clean`   | preserves trusted hints                          | missing terms: Cars, City, Commuting, Fenchurch Street, Nightscape                                                            |
| `mlx-community/Qwen3.5-27B-4bit`                        | `clean`   | preserves trusted hints                          | missing terms: Cars, Cityscape, Commuting, Modern, Nightscape                                                                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `clean`   | preserves trusted hints                          | missing terms: Cars, City, Commuting, Nightscape, Street signs                                                                |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `clean`   | preserves trusted hints                          | missing terms: Cars, Commuting, The Fenchurch Building (The Walki..., Walkie Talkie building, GBR                             |

### `caveat`

| Model                                           | Verdict             | Hint Handling                                    | Key Evidence                                                                                                                                                                                           |
|-------------------------------------------------|---------------------|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`              | `clean`             | preserves trusted hints                          | missing terms: Cars, Cityscape, Commuting, Fenchurch Street, Nightscape                                                                                                                                |
| `mlx-community/nanoLLaVA-1.5-4bit`              | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: Urban, Urban landscape \| low-draft-improvement                                                                                                                                         |
| `mlx-community/LFM2-VL-1.6B-8bit`               | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: formally \| keywords=20 \| low-draft-improvement                                                                                                                                        |
| `mlx-community/LFM2.5-VL-1.6B-bf16`             | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: Architecture, Cars, Commuting, The Fenchurch Building (The Walki..., formally \| keywords=22 \| low-draft-improvement                                                                   |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`      | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: known, formally, 20 \| keywords=20 \| context echo=40% \| low-draft-improvement                                                                                                         |
| `HuggingFaceTB/SmolVLM-Instruct`                | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: The Fenchurch Building (The Walki... \| keywords=20 \| context echo=59% \| low-draft-improvement                                                                                        |
| `mlx-community/SmolVLM-Instruct-bf16`           | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: The Fenchurch Building (The Walki... \| keywords=20 \| context echo=59% \| low-draft-improvement                                                                                        |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` | `model_shortcoming` | preserves trusted hints \| low-draft-improvement | missing sections: title \| missing terms: Cars, Commuting, Street signs, The Fenchurch Building (The Walki..., Urban landscape \| formatting=Unknown tags: &lt;channel\|&gt; \| low-draft-improvement  |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`  | `model_shortcoming` | preserves trusted hints \| low-draft-improvement | missing sections: title \| missing terms: Cars, Commuting, Fenchurch Street, Street signs, The Fenchurch Building (The Walki... \| formatting=Unknown tags: &lt;channel\|&gt; \| low-draft-improvement |
| `mlx-community/Idefics3-8B-Llama3-bf16`         | `model_shortcoming` | degrades trusted hints \| low-draft-improvement  | missing sections: title, description, keywords \| missing terms: Architecture, Cars, City, Cityscape, Commuting \| formatting=Unknown tags: &lt;end_of_utterance&gt; \| low-draft-improvement          |
| `microsoft/Phi-3.5-vision-instruct`             | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: Cars, Commuting, Fenchurch Street, Nightscape, Street signs \| keywords=19 \| low-draft-improvement                                                                                     |
| `mlx-community/Phi-3.5-vision-instruct-bf16`    | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: Cars, Commuting, Fenchurch Street, Nightscape, Street signs \| keywords=19 \| low-draft-improvement                                                                                     |
| `mlx-community/llava-v1.6-mistral-7b-8bit`      | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: The Fenchurch Building (The Walki... \| keywords=20 \| context echo=59% \| low-draft-improvement                                                                                        |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`         | `model_shortcoming` | preserves trusted hints                          | missing sections: title \| missing terms: Commuting, Nightscape, Street signs, The Fenchurch Building (The Walki..., GBR \| thinking trace present                                                     |
| `mlx-community/GLM-4.6V-Flash-6bit`             | `context_budget`    | preserves trusted hints                          | output/prompt=1.93% \| visual input burden=93% \| missing sections: title \| missing terms: Cityscape, Commuting, Fenchurch Street, Nightscape, The Fenchurch Building (The Walki...                   |
| `mlx-community/GLM-4.6V-nvfp4`                  | `context_budget`    | preserves trusted hints                          | output/prompt=1.69% \| visual input burden=93% \| missing sections: title \| missing terms: Commuting, Fenchurch Street, London, Street signs, The Fenchurch Building (The Walki...                    |
| `mlx-community/MolmoPoint-8B-fp16`              | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: Fenchurch Street, The Fenchurch Building (The Walki..., Walkie Talkie building, Fenchurch, Walkie \| low-draft-improvement                                                              |
| `mlx-community/Molmo-7B-D-0924-8bit`            | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: known, formally, 20 \| keywords=20 \| low-draft-improvement                                                                                                                             |
| `mlx-community/Molmo-7B-D-0924-bf16`            | `clean`             | preserves trusted hints \| low-draft-improvement | missing terms: known, formally, 20 \| keywords=20 \| low-draft-improvement                                                                                                                             |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`       | `context_budget`    | ignores trusted hints                            | At mixed burden (16833 tokens), output stayed unusually short (1 tokens; ratio 0.0%; weak text signal empty). \| output/prompt=0.01% \| mixed burden=97%                                               |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`  | `model_shortcoming` | preserves trusted hints                          | missing sections: title \| missing terms: Cityscape, Commuting, Street signs, The Fenchurch Building (The Walki..., GBR \| thinking trace present                                                      |

### `needs_triage`

- None.

### `avoid`

| Model                                               | Verdict             | Hint Handling                                                                                                  | Key Evidence                                                                                                                                                                                       |
|-----------------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Step-3.7-Flash-oQ2e`                 | `runtime_failure`   | not evaluated                                                                                                  | processor error \| model config processor load processor                                                                                                                                           |
| `mlx-community/MiniCPM-V-4.6-8bit`                  | `harness`           | preserves trusted hints                                                                                        | Special control token &lt;/think&gt; appeared in generated text. \| missing terms: Cars, Commuting, Fenchurch Street, London, Nightscape \| reasoning leak \| text-sanity=gibberish(token_noise)   |
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                               | missing terms: Cars, Commuting, Fenchurch Street, London, Street signs \| low-draft-improvement                                                                                                    |
| `qnguyen3/nanoLLaVA`                                | `model_shortcoming` | degrades trusted hints \| low-draft-improvement                                                                | missing sections: keywords \| missing terms: Architecture, Cars, Cityscape, Commuting, Fenchurch Street \| low-draft-improvement                                                                   |
| `mlx-community/FastVLM-0.5B-bf16`                   | `model_shortcoming` | preserves trusted hints \| low-draft-improvement                                                               | missing sections: keywords \| missing terms: Cityscape, Commuting, Modern, Nightscape, Street signs \| low-draft-improvement                                                                       |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `semantic_mismatch` | preserves trusted hints                                                                                        | missing terms: Cityscape, Commuting, Fenchurch Street, Illuminated, Nightscape                                                                                                                     |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | `model_shortcoming` | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement                                                            |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `semantic_mismatch` | degrades trusted hints                                                                                         | missing terms: Cars, City, Commuting, Fenchurch Street, London                                                                                                                                     |
| `mlx-community/pixtral-12b-8bit`                    | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                               | missing terms: Cars, Commuting, Fenchurch Street, London, Street signs \| low-draft-improvement                                                                                                    |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `semantic_mismatch` | degrades trusted hints                                                                                         | missing terms: Cars, City, Commuting, Fenchurch Street, London                                                                                                                                     |
| `mlx-community/gemma-3n-E2B-4bit`                   | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City \| repetitive token=phrase: "do not, and do..."              |
| `mlx-community/InternVL3-14B-8bit`                  | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                               | missing terms: Cars, Commuting, Fenchurch Street, Street signs, The Fenchurch Building (The Walki... \| low-draft-improvement                                                                      |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | `model_shortcoming` | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement                                                            |
| `mlx-community/pixtral-12b-bf16`                    | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                               | missing terms: Cars, Commuting, Fenchurch Street, London, Modern \| low-draft-improvement                                                                                                          |
| `mlx-community/gemma-3n-E4B-it-bf16`                | `model_shortcoming` | preserves trusted hints \| unverified-context-copy                                                             | missing terms: Commuting, Street signs, The Fenchurch Building (The Walki..., Urban landscape, GBR \| keywords=25 \| unverified-context-copy                                                       |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `model_shortcoming` | preserves trusted hints \| low-draft-improvement                                                               | missing sections: title, description, keywords \| missing terms: known, formally, 20 \| low-draft-improvement                                                                                      |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`           | `cutoff_degraded`   | preserves trusted hints                                                                                        | hit token cap (500) \| missing sections: title \| missing terms: Cars, Commuting, Nightscape, Street signs, The Fenchurch Building (The Walki... \| thinking trace present                         |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                               | hit token cap (500) \| missing terms: Cars, Commuting, Fenchurch Street, Nightscape, Street signs \| keyword duplication=51% \| repetitive token=phrase: "the fenchurch building, london..."       |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `model_shortcoming` | preserves trusted hints \| low-draft-improvement                                                               | missing sections: title, description, keywords \| missing terms: Architecture, Cityscape, Commuting, Fenchurch Street, London \| low-draft-improvement                                             |
| `mlx-community/paligemma2-3b-pt-896-4bit`           | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                               | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Nightscape \| reasoning leak                                                                               |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`     | `cutoff_degraded`   | preserves trusted hints                                                                                        | hit token cap (500) \| missing sections: title \| missing terms: Commuting, Nightscape, Street signs, The Fenchurch Building (The Walki..., Urban landscape \| keywords=24                         |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`  | `cutoff_degraded`   | degrades trusted hints \| low-draft-improvement                                                                | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Cars, City, Cityscape, Commuting \| degeneration=character_loop: '00' repeated               |
| `mlx-community/gemma-4-31b-bf16`                    | `model_shortcoming` | preserves trusted hints                                                                                        | missing sections: title, description, keywords \| missing terms: Architecture, Cityscape, Commuting, Fenchurch Street, London                                                                      |
| `Qwen/Qwen3-VL-2B-Instruct`                         | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | At mixed burden (16822 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`           | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | At mixed burden (16822 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`           | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | At mixed burden (16824 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City |
| `mlx-community/X-Reasoner-7B-8bit`                  | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | At mixed burden (16833 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`     | `cutoff_degraded`   | ignores trusted hints \| missing terms: Architecture, Building, Buildings, Cars, City \| low-draft-improvement | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City \| excessive bullets=26                                      |
| `mlx-community/Ornith-1.0-35B-bf16`                 | `semantic_mismatch` | preserves trusted hints                                                                                        | missing terms: Cars, City, Commuting, Fenchurch Street, Nightscape                                                                                                                                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `semantic_mismatch` | preserves trusted hints                                                                                        | missing terms: Cars, City, Commuting, Fenchurch Street, Nightscape                                                                                                                                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `semantic_mismatch` | degrades trusted hints                                                                                         | missing terms: Cars, City, Commuting, Fenchurch Street, Illuminated                                                                                                                                |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

<!-- markdownlint-disable MD060 -->

| Target                                         | Problem                                               | Evidence Snapshot                                                                                                                                                         | Affected Models                                         | Issue Draft                                                                                                                                                              | Evidence Bundle                                                                                                                                                                               | Fixed When                                                   |
|------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| model configuration / repository               | Processor config is missing image processor           | Processor Error \| phase processor_load \| ValueError                                                                                                                     | 1: `mlx-community/Step-3.7-Flash-oQ2e`                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_model-configuration-repository_model-config-processor-load-processor_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260719T014155Z_001_mlx-community_Step-3.7-Flash-oQ2e_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_3c.json) | Load/generation completes or fails with a narrower owner.    |
| `mlx-vlm`                                      | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;/think&gt; \| prompt=1,223 \| output/prompt=7.11% \| stop=completed                                                               | 1: `mlx-community/MiniCPM-V-4.6-8bit`                   | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_stop-token_001.md)                                                   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260719T014155Z_002_mlx-community_MiniCPM-V-4.6-8bit_mlx_vlm_stop_token_001.json)                    | No leaked stop/control tokens.                               |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short | prompt_tokens=16822, repetitive output \| prompt=16,822 \| output/prompt=2.97% \| mixed burden=97% \| stop=max_tokens \| hit token cap (500) \| 4 model cluster           | 4: `Qwen/Qwen3-VL-2B-Instruct` (+3)                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm-mlx_long-context_001.md)                                             | [4 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260719T014155Z_005_Qwen_Qwen3-VL-2B-Instruct_mlx_vlm_mlx_long_context_001.json)                  | Full and reduced reruns avoid context collapse.              |
| model repository                               | Generated text is mixed-script token-soup             | token cap \| missing sections \| trusted overlap \| prompt=3,502 \| output/prompt=14.28% \| mixed burden=86% \| stop=max_tokens \| hit token cap (500) \| 2 model cluster | 2: `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (+1) | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_model_text-sanity_001.md)                                                    | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260719T014155Z_004_mlx-community_Apriel-1.5-15b-Thinker-6bit-MLX_model_text_sanity_001.json)     | Generated text is readable natural language, not token soup. |
<!-- markdownlint-enable MD060 -->

## Model Verdicts

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
- _Key signals:_ missing terms: Cars, Cityscape, Commuting, Fenchurch Street,
  Nightscape
- _Tokens:_ prompt 675 tok; estimated text 493 tok; estimated non-text 182
  tok; generated 154 tok; requested max 500 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Urban, Urban landscape; low-draft-improvement
- _Tokens:_ prompt 605 tok; estimated text 493 tok; estimated non-text 112
  tok; generated 99 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: formally; keywords=20; low-draft-improvement
- _Tokens:_ prompt 865 tok; estimated text 493 tok; estimated non-text 372
  tok; generated 172 tok; requested max 500 tok; stop reason completed


### `mlx-community/MiniCPM-V-4.6-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; missing terms: Cars, Commuting, Fenchurch Street, London, Nightscape;
  reasoning leak; text-sanity=gibberish(token_noise)
- _Tokens:_ prompt 1223 tok; estimated text 493 tok; estimated non-text 730
  tok; generated 87 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Fenchurch Street, London,
  Street signs; low-draft-improvement
- _Tokens:_ prompt 883 tok; estimated text 493 tok; estimated non-text 390
  tok; generated 86 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Architecture, Cars, Commuting, The Fenchurch
  Building (The Walki..., formally; keywords=22; low-draft-improvement
- _Tokens:_ prompt 865 tok; estimated text 493 tok; estimated non-text 372
  tok; generated 234 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Architecture,
  Cars, Cityscape, Commuting, Fenchurch Street; low-draft-improvement
- _Tokens:_ prompt 605 tok; estimated text 493 tok; estimated non-text 112
  tok; generated 129 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Cityscape,
  Commuting, Modern, Nightscape, Street signs; low-draft-improvement
- _Tokens:_ prompt 609 tok; estimated text 493 tok; estimated non-text 116
  tok; generated 214 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: known, formally, 20; keywords=20; context
  echo=40%; low-draft-improvement
- _Tokens:_ prompt 722 tok; estimated text 493 tok; estimated non-text 229
  tok; generated 120 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: The Fenchurch Building (The Walki...;
  keywords=20; context echo=59%; low-draft-improvement
- _Tokens:_ prompt 1822 tok; estimated text 493 tok; estimated non-text 1329
  tok; generated 115 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: The Fenchurch Building (The Walki...;
  keywords=20; context echo=59%; low-draft-improvement
- _Tokens:_ prompt 1822 tok; estimated text 493 tok; estimated non-text 1329
  tok; generated 115 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cityscape, Commuting, Fenchurch Street,
  Illuminated, Nightscape
- _Tokens:_ prompt 3217 tok; estimated text 493 tok; estimated non-text 2724
  tok; generated 134 tok; requested max 500 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

- _Recommendation:_ use with caveats; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Cars, Commuting,
  Street signs, The Fenchurch Building (The Walki..., Urban landscape;
  formatting=Unknown tags: &lt;channel|&gt;; low-draft-improvement
- _Tokens:_ prompt 879 tok; estimated text 493 tok; estimated non-text 386
  tok; generated 95 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City; low-draft-improvement
- _Tokens:_ prompt 1630 tok; estimated text 493 tok; estimated non-text 1137
  tok; generated 18 tok; requested max 500 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-8bit`

- _Recommendation:_ use with caveats; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Cars, Commuting,
  Fenchurch Street, Street signs, The Fenchurch Building (The Walki...;
  formatting=Unknown tags: &lt;channel|&gt;; low-draft-improvement
- _Tokens:_ prompt 879 tok; estimated text 493 tok; estimated non-text 386
  tok; generated 91 tok; requested max 500 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ use with caveats; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Cars, City, Cityscape, Commuting; formatting=Unknown
  tags: &lt;end_of_utterance&gt;; low-draft-improvement
- _Tokens:_ prompt 2887 tok; estimated text 493 tok; estimated non-text 2394
  tok; generated 26 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Fenchurch Street, Nightscape,
  Street signs; keywords=19; low-draft-improvement
- _Tokens:_ prompt 1436 tok; estimated text 493 tok; estimated non-text 943
  tok; generated 171 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Fenchurch Street, Nightscape,
  Street signs; keywords=19; low-draft-improvement
- _Tokens:_ prompt 1436 tok; estimated text 493 tok; estimated non-text 943
  tok; generated 171 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Nightscape, The Fenchurch Building (The
  Walki..., GBR, known, formally; low-draft-improvement
- _Tokens:_ prompt 2388 tok; estimated text 493 tok; estimated non-text 1895
  tok; generated 92 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Fenchurch Street,
  London
- _Tokens:_ prompt 3218 tok; estimated text 493 tok; estimated non-text 2725
  tok; generated 106 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Fenchurch Street, London,
  Street signs; low-draft-improvement
- _Tokens:_ prompt 3411 tok; estimated text 493 tok; estimated non-text 2918
  tok; generated 74 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Fenchurch Street,
  London
- _Tokens:_ prompt 3218 tok; estimated text 493 tok; estimated non-text 2725
  tok; generated 121 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Building, Buildings, Cars, City;
  repetitive token=phrase: "do not, and do..."
- _Tokens:_ prompt 869 tok; estimated text 493 tok; estimated non-text 376
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: The Fenchurch Building (The Walki...;
  keywords=20; context echo=59%; low-draft-improvement
- _Tokens:_ prompt 2834 tok; estimated text 493 tok; estimated non-text 2341
  tok; generated 124 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Fenchurch Street, Street
  signs, The Fenchurch Building (The Walki...; low-draft-improvement
- _Tokens:_ prompt 2388 tok; estimated text 493 tok; estimated non-text 1895
  tok; generated 87 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City; low-draft-improvement
- _Tokens:_ prompt 1630 tok; estimated text 493 tok; estimated non-text 1137
  tok; generated 17 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Fenchurch Street, Street
  signs, The Fenchurch Building (The Walki...; low-draft-improvement
- _Tokens:_ prompt 878 tok; estimated text 493 tok; estimated non-text 385
  tok; generated 104 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Street signs, The Fenchurch
  Building (The Walki..., GBR; low-draft-improvement
- _Tokens:_ prompt 883 tok; estimated text 493 tok; estimated non-text 390
  tok; generated 92 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ use with caveats; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Commuting,
  Nightscape, Street signs, The Fenchurch Building (The Walki..., GBR;
  thinking trace present
- _Tokens:_ prompt 1598 tok; estimated text 493 tok; estimated non-text 1105
  tok; generated 302 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Fenchurch Street, London,
  Modern; low-draft-improvement
- _Tokens:_ prompt 3411 tok; estimated text 493 tok; estimated non-text 2918
  tok; generated 84 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Fenchurch Street, Street
  signs, The Fenchurch Building (The Walki...; low-draft-improvement
- _Tokens:_ prompt 878 tok; estimated text 493 tok; estimated non-text 385
  tok; generated 96 tok; requested max 500 tok; stop reason completed


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cityscape, Commuting, London, Nightscape, The
  Fenchurch Building (The Walki...; low-draft-improvement
- _Tokens:_ prompt 2685 tok; estimated text 493 tok; estimated non-text 2192
  tok; generated 112 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Run a controlled reduced-image or lower-visual-token comparison
  before assigning the context-boundary behaviour to mlx, mlx-vlm, or the
  model.
- _Key signals:_ output/prompt=1.93%; visual input burden=93%; missing
  sections: title; missing terms: Cityscape, Commuting, Fenchurch Street,
  Nightscape, The Fenchurch Building (The Walki...
- _Tokens:_ prompt 6627 tok; estimated text 493 tok; estimated non-text 6134
  tok; generated 128 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, Street signs, The Fenchurch
  Building (The Walki..., Urban landscape, GBR; keywords=25;
  unverified-context-copy
- _Tokens:_ prompt 877 tok; estimated text 493 tok; estimated non-text 384
  tok; generated 346 tok; requested max 500 tok; stop reason completed


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: known, formally, 20; low-draft-improvement
- _Tokens:_ prompt 575 tok; estimated text 493 tok; estimated non-text 82 tok;
  generated 147 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  Cars, Commuting, Nightscape, Street signs, The Fenchurch Building (The
  Walki...; thinking trace present
- _Tokens:_ prompt 1598 tok; estimated text 493 tok; estimated non-text 1105
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: Cars, Commuting,
  Fenchurch Street, Nightscape, Street signs; keyword duplication=51%;
  repetitive token=phrase: "the fenchurch building, london..."
- _Tokens:_ prompt 6627 tok; estimated text 493 tok; estimated non-text 6134
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Cityscape, Commuting, Fenchurch Street, London;
  low-draft-improvement
- _Tokens:_ prompt 576 tok; estimated text 493 tok; estimated non-text 83 tok;
  generated 51 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Nightscape; reasoning leak
- _Tokens:_ prompt 4702 tok; estimated text 493 tok; estimated non-text 4209
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  Commuting, Nightscape, Street signs, The Fenchurch Building (The Walki...,
  Urban landscape; keywords=24
- _Tokens:_ prompt 3502 tok; estimated text 493 tok; estimated non-text 3009
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Cars, City, Cityscape, Commuting;
  degeneration=character_loop: '00' repeated
- _Tokens:_ prompt 1927 tok; estimated text 493 tok; estimated non-text 1434
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Cityscape, Commuting, Fenchurch Street, London
- _Tokens:_ prompt 871 tok; estimated text 493 tok; estimated non-text 378
  tok; generated 106 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Run a controlled reduced-image or lower-visual-token comparison
  before assigning the context-boundary behaviour to mlx, mlx-vlm, or the
  model.
- _Key signals:_ output/prompt=1.69%; visual input burden=93%; missing
  sections: title; missing terms: Commuting, Fenchurch Street, London, Street
  signs, The Fenchurch Building (The Walki...
- _Tokens:_ prompt 6627 tok; estimated text 493 tok; estimated non-text 6134
  tok; generated 112 tok; requested max 500 tok; stop reason completed


### `Qwen/Qwen3-VL-2B-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16822 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City
- _Tokens:_ prompt 16822 tok; estimated text 493 tok; estimated non-text 16329
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16822 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City
- _Tokens:_ prompt 16822 tok; estimated text 493 tok; estimated non-text 16329
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Fenchurch Street, The Fenchurch Building (The
  Walki..., Walkie Talkie building, Fenchurch, Walkie; low-draft-improvement
- _Tokens:_ prompt 3420 tok; estimated text 493 tok; estimated non-text 2927
  tok; generated 108 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16824 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City
- _Tokens:_ prompt 16824 tok; estimated text 493 tok; estimated non-text 16331
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: known, formally, 20; keywords=20;
  low-draft-improvement
- _Tokens:_ prompt 1792 tok; estimated text 493 tok; estimated non-text 1299
  tok; generated 140 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: known, formally, 20; keywords=20;
  low-draft-improvement
- _Tokens:_ prompt 1792 tok; estimated text 493 tok; estimated non-text 1299
  tok; generated 140 tok; requested max 500 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16833 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City
- _Tokens:_ prompt 16833 tok; estimated text 493 tok; estimated non-text 16340
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Building, Buildings, Cars, City;
  excessive bullets=26
- _Tokens:_ prompt 1630 tok; estimated text 493 tok; estimated non-text 1137
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Ornith-1.0-35B-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Fenchurch Street,
  Nightscape
- _Tokens:_ prompt 16852 tok; estimated text 493 tok; estimated non-text 16359
  tok; generated 120 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Fenchurch Street, The
  Fenchurch Building (The Walki..., GBR
- _Tokens:_ prompt 16852 tok; estimated text 493 tok; estimated non-text 16359
  tok; generated 110 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Fenchurch Street,
  Nightscape
- _Tokens:_ prompt 16852 tok; estimated text 493 tok; estimated non-text 16359
  tok; generated 114 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Fenchurch Street,
  Nightscape
- _Tokens:_ prompt 16852 tok; estimated text 493 tok; estimated non-text 16359
  tok; generated 111 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Run a controlled reduced-image or lower-visual-token comparison
  before assigning the context-boundary behaviour to mlx, mlx-vlm, or the
  model.
- _Key signals:_ At mixed burden (16833 tokens), output stayed unusually short
  (1 tokens; ratio 0.0%; weak text signal empty).; output/prompt=0.01%; mixed
  burden=97%
- _Tokens:_ prompt 16833 tok; estimated text 493 tok; estimated non-text 16340
  tok; generated 1 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Fenchurch Street,
  Illuminated
- _Tokens:_ prompt 16852 tok; estimated text 493 tok; estimated non-text 16359
  tok; generated 112 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ use with caveats; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Cityscape, Commuting,
  Street signs, The Fenchurch Building (The Walki..., GBR; thinking trace
  present
- _Tokens:_ prompt 1598 tok; estimated text 493 tok; estimated non-text 1105
  tok; generated 341 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Cityscape, Commuting, Modern, Nightscape
- _Tokens:_ prompt 16852 tok; estimated text 493 tok; estimated non-text 16359
  tok; generated 118 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Nightscape, Street
  signs
- _Tokens:_ prompt 16852 tok; estimated text 493 tok; estimated non-text 16359
  tok; generated 144 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, The Fenchurch Building (The
  Walki..., Walkie Talkie building, GBR
- _Tokens:_ prompt 16852 tok; estimated text 493 tok; estimated non-text 16359
  tok; generated 128 tok; requested max 500 tok; stop reason completed

