<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

Generated on: 2026-07-17 23:16:01 BST

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Companion artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (88/100) | Desc 83 | Keywords 76 | Δ+14 | 62.8 tps
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🏆 A (84/100) | Desc 90 | Keywords 78 | Δ+10 | 119.0 tps
- `mlx-community/Ornith-1.0-35B-bf16`: 🏆 A (83/100) | Desc 99 | Keywords 55 | Δ+10 | 51.4 tps
- `mlx-community/Qwen3.5-35B-A3B-4bit`: 🏆 A (82/100) | Desc 81 | Keywords 58 | Δ+9 | 94.5 tps
- `mlx-community/Molmo-7B-D-0924-bf16`: ✅ B (77/100) | Desc 84 | Keywords 52 | Δ+4 | 29.2 tps

### Watchlist

- `mlx-community/gemma-4-31b-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-74 | 9.1 tps | context ignored, harness
- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-74 | 179.1 tps | context ignored, harness, long context
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-74 | 139.1 tps | context ignored, harness, long context
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (6/100) | Desc 43 | Keywords 0 | Δ-68 | 68.5 tps | context ignored, harness, long context
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-53 | 73.6 tps | context ignored, cutoff, degeneration, generation loop, harness, long context, missing sections, repetitive, text sanity

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                                   | Verdict   | Hint Handling                                    | Key Evidence                                                                       |
|---------------------------------------------------------|-----------|--------------------------------------------------|------------------------------------------------------------------------------------|
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Cars, Swimming, Walking, Water, Waterfront \| low-draft-improvement |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Cars, Deal, Kent, Promenade, Sitting \| low-draft-improvement       |
| `mlx-community/gemma-4-31b-it-4bit`                     | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Sitting, Swimming, Walking, GBR, showing \| low-draft-improvement   |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Deal, Kent, Sitting, Swimming, Walking \| low-draft-improvement     |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `clean`   | preserves trusted hints                          | missing terms: Kent, Sitting, Swimming, Walking, GBR                               |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `clean`   | preserves trusted hints                          | missing terms: Horizon, Kent, Sitting, Swimming, Walking                           |
| `mlx-community/Qwen3.5-27B-4bit`                        | `clean`   | preserves trusted hints                          | missing terms: Seaside, Shore, Walking, Water, architecture                        |

### `caveat`

| Model                                        | Verdict          | Hint Handling                                                                                            | Key Evidence                                                                                                                                                                                                |
|----------------------------------------------|------------------|----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`           | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | keywords=20 \| low-draft-improvement                                                                                                                                                                        |
| `mlx-community/LFM2.5-VL-1.6B-bf16`          | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | missing terms: Swimming, Walking, Water, Waterfront \| low-draft-improvement                                                                                                                                |
| `microsoft/Phi-3.5-vision-instruct`          | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | missing terms: Swimming, GBR, showing \| keywords=22 \| low-draft-improvement                                                                                                                               |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | missing terms: Swimming, GBR, showing \| keywords=22 \| low-draft-improvement                                                                                                                               |
| `mlx-community/pixtral-12b-8bit`             | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | missing terms: Cars, Deal, Horizon, Kent, Landscape \| keywords=19 \| low-draft-improvement                                                                                                                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`   | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | keywords=20 \| low-draft-improvement                                                                                                                                                                        |
| `mlx-community/Idefics3-8B-Llama3-bf16`      | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | keywords=20 \| context echo=60% \| formatting=Unknown tags: &lt;end_of_utterance&gt; \| low-draft-improvement                                                                                               |
| `mlx-community/gemma-3-27b-it-qat-4bit`      | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | missing terms: Cars, Kent, Sitting, Swimming, Walking \| keywords=20 \| low-draft-improvement                                                                                                               |
| `mlx-community/InternVL3-14B-8bit`           | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | missing terms: Deal, Kent, Sitting, GBR, view \| low-draft-improvement                                                                                                                                      |
| `mlx-community/GLM-4.6V-Flash-6bit`          | `context_budget` | preserves trusted hints \| low-draft-improvement                                                         | output/prompt=1.33% \| visual input burden=93% \| missing sections: title \| missing terms: Cars, Horizon, Swimming, Walking, GBR                                                                           |
| `mlx-community/gemma-3-27b-it-qat-8bit`      | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | missing terms: Cars, Deal, Kent, Sitting, Swimming \| keywords=19 \| low-draft-improvement                                                                                                                  |
| `Qwen/Qwen3-VL-2B-Instruct`                  | `context_budget` | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement | Output appears truncated to about 2 tokens. \| At mixed burden (16898 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated). \| output/prompt=0.01% \| mixed burden=97% |
| `mlx-community/GLM-4.6V-nvfp4`               | `context_budget` | preserves trusted hints \| low-draft-improvement                                                         | output/prompt=1.41% \| visual input burden=93% \| missing sections: title \| missing terms: Kent, Swimming, Walking, GBR, view                                                                              |
| `mlx-community/Molmo-7B-D-0924-8bit`         | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | missing terms: Seafront, GBR, view, showing, seafront \| keywords=19 \| low-draft-improvement                                                                                                               |
| `mlx-community/X-Reasoner-7B-8bit`           | `context_budget` | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement | Output appears truncated to about 5 tokens. \| At mixed burden (16909 tokens), output stayed unusually short (5 tokens; ratio 0.0%; weak text signal truncated). \| output/prompt=0.03% \| mixed burden=97% |
| `mlx-community/Molmo-7B-D-0924-bf16`         | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | missing terms: Horizon \| keywords=23 \| low-draft-improvement                                                                                                                                              |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`    | `context_budget` | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement | Output appears truncated to about 2 tokens. \| At mixed burden (16898 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated). \| output/prompt=0.01% \| mixed burden=97% |
| `mlx-community/Qwen3.5-9B-MLX-4bit`          | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | missing terms: Walking \| keywords=19 \| low-draft-improvement                                                                                                                                              |
| `mlx-community/Qwen3.5-35B-A3B-4bit`         | `clean`          | preserves trusted hints                                                                                  | missing terms: Deal, Horizon, Kent, Swimming, Walking                                                                                                                                                       |
| `mlx-community/Qwen3.6-27B-mxfp8`            | `clean`          | preserves trusted hints                                                                                  | missing terms: Kent, Promenade, Seaside, Swimming, Walking \| keywords=19                                                                                                                                   |
| `mlx-community/Ornith-1.0-35B-bf16`          | `clean`          | preserves trusted hints                                                                                  | missing terms: Cars, Deal, Swimming, architecture, GBR \| keywords=20                                                                                                                                       |
| `mlx-community/Qwen3.5-27B-mxfp8`            | `clean`          | preserves trusted hints \| low-draft-improvement                                                         | missing terms: Kent, Sitting, Walking, GBR, view \| keywords=20 \| low-draft-improvement                                                                                                                    |

### `needs_triage`

- None.

### `avoid`

| Model                                               | Verdict             | Hint Handling                                                                                               | Key Evidence                                                                                                                                                                                 |
|-----------------------------------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `model_shortcoming` | preserves trusted hints \| low-draft-improvement                                                            | missing sections: keywords \| missing terms: Cars, Coastline, Horizon, Landscape, People \| low-draft-improvement                                                                            |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `model_shortcoming` | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Horizon \| low-draft-improvement | missing sections: title, description, keywords \| missing terms: Beach, Buildings, Cars, Coastline, Horizon \| low-draft-improvement                                                         |
| `mlx-community/SmolVLM-Instruct-bf16`               | `model_shortcoming` | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Horizon \| low-draft-improvement | missing sections: title, description, keywords \| missing terms: Beach, Buildings, Cars, Coastline, Horizon \| low-draft-improvement                                                         |
| `mlx-community/MiniCPM-V-4.6-8bit`                  | `harness`           | preserves trusted hints \| low-draft-improvement                                                            | Special control token &lt;/think&gt; appeared in generated text. \| missing terms: Cars, Coastline, Deal, Horizon, Kent \| reasoning leak \| text-sanity=gibberish(token_noise)              |
| `mlx-community/FastVLM-0.5B-bf16`                   | `model_shortcoming` | preserves trusted hints \| low-draft-improvement                                                            | missing sections: keywords \| missing terms: Horizon, Landscape, Promenade, Sitting, Swimming \| low-draft-improvement                                                                       |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | `model_shortcoming` | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement    | missing sections: title, description, keywords \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement                                                            |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `semantic_mismatch` | preserves trusted hints                                                                                     | missing terms: Cars, Coastline, Deal, Horizon, Kent                                                                                                                                          |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`     | `model_shortcoming` | preserves trusted hints \| low-draft-improvement                                                            | missing sections: title \| missing terms: Sitting, Swimming, Walking, GBR, showing \| formatting=Unknown tags: &lt;channel\|&gt; \| low-draft-improvement                                    |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`      | `model_shortcoming` | preserves trusted hints \| low-draft-improvement                                                            | missing sections: title \| missing terms: Coastline, Horizon, Shore, Sitting, Swimming \| formatting=Unknown tags: &lt;channel\|&gt; \| low-draft-improvement                                |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`          | `model_shortcoming` | preserves trusted hints \| unverified-context-copy                                                          | keyword duplication=40% \| unverified-context-copy                                                                                                                                           |
| `mlx-community/InternVL3-8B-bf16`                   | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                            | missing terms: Deal, Kent, Sitting, Swimming, Seafront \| low-draft-improvement                                                                                                              |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `semantic_mismatch` | preserves trusted hints                                                                                     | missing terms: Coastline, Deal, Kent, Landscape, Promenade                                                                                                                                   |
| `mlx-community/gemma-4-31b-bf16`                    | `harness`           | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement    | Output appears truncated to about 4 tokens. \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement                                                               |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `semantic_mismatch` | preserves trusted hints                                                                                     | missing terms: Cars, Coastline, Deal, Kent, Landscape                                                                                                                                        |
| `mlx-community/gemma-3n-E2B-4bit`                   | `cutoff_degraded`   | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement    | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| excessive bullets=63                                      |
| `qnguyen3/nanoLLaVA`                                | `cutoff_degraded`   | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Horizon \| low-draft-improvement | hit token cap (500) \| missing sections: description, keywords \| missing terms: Beach, Buildings, Cars, Coastline, Horizon \| context echo=100%                                             |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | `model_shortcoming` | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement    | missing sections: title, description, keywords \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement                                                            |
| `mlx-community/pixtral-12b-bf16`                    | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                            | missing terms: Cars, Deal, Horizon, Kent, Landscape \| keywords=19 \| low-draft-improvement                                                                                                  |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `model_shortcoming` | preserves trusted hints \| low-draft-improvement                                                            | missing sections: title, description, keywords \| missing terms: GBR \| low-draft-improvement                                                                                                |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `cutoff_degraded`   | preserves trusted hints                                                                                     | hit token cap (500) \| missing sections: title \| missing terms: showing, day \| keywords=61                                                                                                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`           | `cutoff_degraded`   | preserves trusted hints                                                                                     | hit token cap (500) \| missing sections: title \| missing terms: Water, showing \| thinking trace present                                                                                    |
| `mlx-community/gemma-3n-E4B-it-bf16`                | `model_shortcoming` | preserves trusted hints                                                                                     | missing sections: title, description, keywords \| missing terms: Landscape, Swimming, Waterfront, architecture, GBR                                                                          |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                | `cutoff_degraded`   | preserves trusted hints                                                                                     | hit token cap (500) \| missing terms: GBR \| keywords=196                                                                                                                                    |
| `mlx-community/paligemma2-3b-pt-896-4bit`           | `cutoff_degraded`   | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement    | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| excessive bullets=44                                      |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`     | `cutoff_degraded`   | preserves trusted hints                                                                                     | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Coastline, Horizon, Landscape, Seaside, Shore \| reasoning leak                                      |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`  | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                            | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Cars, Coastline, Horizon, Landscape, People \| repetitive token=phrase: "stade stade stade stade..." |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `model_shortcoming` | degrades trusted hints \| low-draft-improvement                                                             | missing sections: title, description, keywords \| missing terms: Cars, Coastline, Horizon, Kent, Landscape \| low-draft-improvement                                                          |
| `mlx-community/MolmoPoint-8B-fp16`                  | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                            | missing terms: Cars, Coastline, Deal, Horizon, Kent \| keywords=22 \| low-draft-improvement                                                                                                  |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`     | `cutoff_degraded`   | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement    | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement                                     |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`           | `cutoff_degraded`   | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Deal \| low-draft-improvement    | At mixed burden (16900 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Beach, Buildings, Cars, Coastline, Deal |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`           | `cutoff_degraded`   | ignores trusted hints \| missing terms: Beach, Buildings, Cars, Coastline, Horizon \| low-draft-improvement | At mixed burden (16909 tokens), output became repetitive. \| hit token cap (500) \| missing sections: description \| missing terms: Beach, Buildings, Cars, Coastline, Horizon               |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `cutoff_degraded`   | preserves trusted hints                                                                                     | hit token cap (500) \| missing sections: title \| missing terms: Cars, Coastline, Landscape, Promenade, Seaside \| keyword duplication=52%                                                   |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                                    | Affected Models                                   | Issue Draft                                                                                                                              | Evidence Bundle                                                                                                                                                                            | Fixed When                                          |
|----------------------------------------------------------|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;/think&gt; \| prompt=1,205 \| output/prompt=6.56% \| stop=completed                                                                                          | 1: `mlx-community/MiniCPM-V-4.6-8bit`             | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx-vlm_stop-token_001.md)                   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T221600Z_001_mlx-community_MiniCPM-V-4.6-8bit_mlx_vlm_stop_token_001.json)                 | No leaked stop/control tokens.                      |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~4 \| prompt=853 \| output/prompt=0.47% \| stop=completed                                                                                                                            | 1: `mlx-community/gemma-4-31b-bf16`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_model-config-mlx-vlm_prompt-template_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T221600Z_002_mlx-community_gemma-4-31b-bf16_model_config_mlx_vlm_prompt_template_001.json) | Requested sections render without template leakage. |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | generated_tokens~2 \| prompt_tokens=16898, output_tokens=2, output/prompt=0.0%, weak text=truncated \| prompt=16,898 \| output/prompt=0.01% \| mixed burden=97% \| stop=completed \| 3 model cluster | 3: `Qwen/Qwen3-VL-2B-Instruct` (+2)               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm-mlx_long-context_001.md)             | [3 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T221600Z_003_Qwen_Qwen3-VL-2B-Instruct_mlx_vlm_mlx_long_context_001.json)               | Full and reduced reruns avoid context collapse.     |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | prompt_tokens=16909, repetitive output \| prompt=16,909 \| output/prompt=2.96% \| mixed burden=97% \| stop=max_tokens \| hit token cap (500) \| 2 model cluster                                      | 2: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+1) | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm-mlx_long-context_002.md)             | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T221600Z_007_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_002.json) | Full and reduced reruns avoid context collapse.     |
<!-- markdownlint-enable MD060 -->

## Model Verdicts

### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; low-draft-improvement
- _Tokens:_ prompt 657 tok; estimated text 488 tok; estimated non-text 169
  tok; generated 100 tok; requested max 500 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Cars, Coastline,
  Horizon, Landscape, People; low-draft-improvement
- _Tokens:_ prompt 585 tok; estimated text 488 tok; estimated non-text 97 tok;
  generated 47 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Swimming, Walking, Water, Waterfront;
  low-draft-improvement
- _Tokens:_ prompt 843 tok; estimated text 488 tok; estimated non-text 355
  tok; generated 88 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Beach, Buildings, Cars, Coastline, Horizon; low-draft-improvement
- _Tokens:_ prompt 1800 tok; estimated text 488 tok; estimated non-text 1312
  tok; generated 15 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Beach, Buildings, Cars, Coastline, Horizon; low-draft-improvement
- _Tokens:_ prompt 1800 tok; estimated text 488 tok; estimated non-text 1312
  tok; generated 15 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Swimming, Walking, Water, Waterfront;
  low-draft-improvement
- _Tokens:_ prompt 843 tok; estimated text 488 tok; estimated non-text 355
  tok; generated 125 tok; requested max 500 tok; stop reason completed


### `mlx-community/MiniCPM-V-4.6-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; missing terms: Cars, Coastline, Deal, Horizon, Kent; reasoning leak;
  text-sanity=gibberish(token_noise)
- _Tokens:_ prompt 1205 tok; estimated text 488 tok; estimated non-text 717
  tok; generated 79 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Deal, Kent, Promenade, Sitting;
  low-draft-improvement
- _Tokens:_ prompt 865 tok; estimated text 488 tok; estimated non-text 377
  tok; generated 89 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Horizon,
  Landscape, Promenade, Sitting, Swimming; low-draft-improvement
- _Tokens:_ prompt 589 tok; estimated text 488 tok; estimated non-text 101
  tok; generated 214 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Beach, Buildings, Cars, Coastline, Deal; low-draft-improvement
- _Tokens:_ prompt 1612 tok; estimated text 488 tok; estimated non-text 1124
  tok; generated 17 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Coastline, Deal, Horizon, Kent
- _Tokens:_ prompt 2845 tok; estimated text 488 tok; estimated non-text 2357
  tok; generated 118 tok; requested max 500 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Sitting, Swimming,
  Walking, GBR, showing; formatting=Unknown tags: &lt;channel|&gt;;
  low-draft-improvement
- _Tokens:_ prompt 861 tok; estimated text 488 tok; estimated non-text 373
  tok; generated 87 tok; requested max 500 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Coastline, Horizon,
  Shore, Sitting, Swimming; formatting=Unknown tags: &lt;channel|&gt;;
  low-draft-improvement
- _Tokens:_ prompt 861 tok; estimated text 488 tok; estimated non-text 373
  tok; generated 100 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keyword duplication=40%; unverified-context-copy
- _Tokens:_ prompt 701 tok; estimated text 488 tok; estimated non-text 213
  tok; generated 308 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Swimming, GBR, showing; keywords=22;
  low-draft-improvement
- _Tokens:_ prompt 1424 tok; estimated text 488 tok; estimated non-text 936
  tok; generated 162 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Swimming, GBR, showing; keywords=22;
  low-draft-improvement
- _Tokens:_ prompt 1424 tok; estimated text 488 tok; estimated non-text 936
  tok; generated 162 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Deal, Kent, Sitting, Swimming, Seafront;
  low-draft-improvement
- _Tokens:_ prompt 2880 tok; estimated text 488 tok; estimated non-text 2392
  tok; generated 79 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Coastline, Deal, Kent, Landscape, Promenade
- _Tokens:_ prompt 2846 tok; estimated text 488 tok; estimated non-text 2358
  tok; generated 118 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Deal, Horizon, Kent, Landscape;
  keywords=19; low-draft-improvement
- _Tokens:_ prompt 2917 tok; estimated text 488 tok; estimated non-text 2429
  tok; generated 93 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 4 tokens.; missing terms:
  Beach, Buildings, Cars, Coastline, Deal; low-draft-improvement
- _Tokens:_ prompt 853 tok; estimated text 488 tok; estimated non-text 365
  tok; generated 4 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Coastline, Deal, Kent, Landscape
- _Tokens:_ prompt 2846 tok; estimated text 488 tok; estimated non-text 2358
  tok; generated 113 tok; requested max 500 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; low-draft-improvement
- _Tokens:_ prompt 2608 tok; estimated text 488 tok; estimated non-text 2120
  tok; generated 109 tok; requested max 500 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; context echo=60%; formatting=Unknown tags:
  &lt;end_of_utterance&gt;; low-draft-improvement
- _Tokens:_ prompt 2866 tok; estimated text 488 tok; estimated non-text 2378
  tok; generated 94 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Beach, Buildings, Cars, Coastline, Deal; excessive
  bullets=63
- _Tokens:_ prompt 847 tok; estimated text 488 tok; estimated non-text 359
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Kent, Sitting, Swimming, Walking;
  keywords=20; low-draft-improvement
- _Tokens:_ prompt 856 tok; estimated text 488 tok; estimated non-text 368
  tok; generated 95 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: description, keywords;
  missing terms: Beach, Buildings, Cars, Coastline, Horizon; context echo=100%
- _Tokens:_ prompt 585 tok; estimated text 488 tok; estimated non-text 97 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Deal, Kent, Sitting, GBR, view;
  low-draft-improvement
- _Tokens:_ prompt 2880 tok; estimated text 488 tok; estimated non-text 2392
  tok; generated 93 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Beach, Buildings, Cars, Coastline, Deal; low-draft-improvement
- _Tokens:_ prompt 1612 tok; estimated text 488 tok; estimated non-text 1124
  tok; generated 17 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Sitting, Swimming, Walking, GBR, showing;
  low-draft-improvement
- _Tokens:_ prompt 865 tok; estimated text 488 tok; estimated non-text 377
  tok; generated 95 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Deal, Horizon, Kent, Landscape;
  keywords=19; low-draft-improvement
- _Tokens:_ prompt 2917 tok; estimated text 488 tok; estimated non-text 2429
  tok; generated 89 tok; requested max 500 tok; stop reason completed


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Deal, Kent, Sitting, Swimming, Walking;
  low-draft-improvement
- _Tokens:_ prompt 2313 tok; estimated text 488 tok; estimated non-text 1825
  tok; generated 97 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a visual input burden issue first; reduce that
  input load or inspect long-context handling before judging output quality.
- _Key signals:_ output/prompt=1.33%; visual input burden=93%; missing
  sections: title; missing terms: Cars, Horizon, Swimming, Walking, GBR
- _Tokens:_ prompt 6593 tok; estimated text 488 tok; estimated non-text 6105
  tok; generated 88 tok; requested max 500 tok; stop reason completed


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: GBR; low-draft-improvement
- _Tokens:_ prompt 555 tok; estimated text 488 tok; estimated non-text 67 tok;
  generated 102 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Deal, Kent, Sitting, Swimming;
  keywords=19; low-draft-improvement
- _Tokens:_ prompt 856 tok; estimated text 488 tok; estimated non-text 368
  tok; generated 105 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  showing, day; keywords=61
- _Tokens:_ prompt 1571 tok; estimated text 488 tok; estimated non-text 1083
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  Water, showing; thinking trace present
- _Tokens:_ prompt 1571 tok; estimated text 488 tok; estimated non-text 1083
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Landscape, Swimming, Waterfront, architecture, GBR
- _Tokens:_ prompt 855 tok; estimated text 488 tok; estimated non-text 367
  tok; generated 401 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: GBR; keywords=196
- _Tokens:_ prompt 6593 tok; estimated text 488 tok; estimated non-text 6105
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Beach, Buildings, Cars, Coastline, Deal; excessive
  bullets=44
- _Tokens:_ prompt 4684 tok; estimated text 488 tok; estimated non-text 4196
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `Qwen/Qwen3-VL-2B-Instruct`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a mixed burden issue first; reduce that input
  load or inspect long-context handling before judging output quality.
- _Key signals:_ Output appears truncated to about 2 tokens.; At mixed burden
  (16898 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak
  text signal truncated).; output/prompt=0.01%; mixed burden=97%
- _Tokens:_ prompt 16898 tok; estimated text 488 tok; estimated non-text 16410
  tok; generated 2 tok; requested max 500 tok; stop reason completed


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Coastline, Horizon, Landscape, Seaside, Shore;
  reasoning leak
- _Tokens:_ prompt 3008 tok; estimated text 488 tok; estimated non-text 2520
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Cars, Coastline, Horizon, Landscape, People;
  repetitive token=phrase: "stade stade stade stade..."
- _Tokens:_ prompt 1887 tok; estimated text 488 tok; estimated non-text 1399
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a visual input burden issue first; reduce that
  input load or inspect long-context handling before judging output quality.
- _Key signals:_ output/prompt=1.41%; visual input burden=93%; missing
  sections: title; missing terms: Kent, Swimming, Walking, GBR, view
- _Tokens:_ prompt 6593 tok; estimated text 488 tok; estimated non-text 6105
  tok; generated 93 tok; requested max 500 tok; stop reason completed


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Cars, Coastline, Horizon, Kent, Landscape; low-draft-improvement
- _Tokens:_ prompt 556 tok; estimated text 488 tok; estimated non-text 68 tok;
  generated 70 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Seafront, GBR, view, showing, seafront;
  keywords=19; low-draft-improvement
- _Tokens:_ prompt 1764 tok; estimated text 488 tok; estimated non-text 1276
  tok; generated 114 tok; requested max 500 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a mixed burden issue first; reduce that input
  load or inspect long-context handling before judging output quality.
- _Key signals:_ Output appears truncated to about 5 tokens.; At mixed burden
  (16909 tokens), output stayed unusually short (5 tokens; ratio 0.0%; weak
  text signal truncated).; output/prompt=0.03%; mixed burden=97%
- _Tokens:_ prompt 16909 tok; estimated text 488 tok; estimated non-text 16421
  tok; generated 5 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Horizon; keywords=23; low-draft-improvement
- _Tokens:_ prompt 1764 tok; estimated text 488 tok; estimated non-text 1276
  tok; generated 173 tok; requested max 500 tok; stop reason completed


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Coastline, Deal, Horizon, Kent;
  keywords=22; low-draft-improvement
- _Tokens:_ prompt 3381 tok; estimated text 488 tok; estimated non-text 2893
  tok; generated 119 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a mixed burden issue first; reduce that input
  load or inspect long-context handling before judging output quality.
- _Key signals:_ Output appears truncated to about 2 tokens.; At mixed burden
  (16898 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak
  text signal truncated).; output/prompt=0.01%; mixed burden=97%
- _Tokens:_ prompt 16898 tok; estimated text 488 tok; estimated non-text 16410
  tok; generated 2 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Beach, Buildings, Cars, Coastline, Deal;
  low-draft-improvement
- _Tokens:_ prompt 1612 tok; estimated text 488 tok; estimated non-text 1124
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16900 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Beach, Buildings, Cars, Coastline, Deal
- _Tokens:_ prompt 16900 tok; estimated text 488 tok; estimated non-text 16412
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Walking; keywords=19; low-draft-improvement
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 95 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Deal, Horizon, Kent, Swimming, Walking
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 121 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Kent, Sitting, Swimming, Walking, GBR
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 105 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Horizon, Kent, Sitting, Swimming, Walking
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 110 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Kent, Promenade, Seaside, Swimming, Walking;
  keywords=19
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 124 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ornith-1.0-35B-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Deal, Swimming, architecture, GBR;
  keywords=20
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 120 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Kent, Sitting, Walking, GBR, view;
  keywords=20; low-draft-improvement
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 116 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16909 tokens), output became repetitive.;
  hit token cap (500); missing sections: description; missing terms: Beach,
  Buildings, Cars, Coastline, Horizon
- _Tokens:_ prompt 16909 tok; estimated text 488 tok; estimated non-text 16421
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Seaside, Shore, Walking, Water, architecture
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 112 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  Cars, Coastline, Landscape, Promenade, Seaside; keyword duplication=52%
- _Tokens:_ prompt 1571 tok; estimated text 488 tok; estimated non-text 1083
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

