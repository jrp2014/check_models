<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

Generated on: 2026-07-10 23:37:01 BST

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Companion artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Qwen3.5-27B-mxfp8`: 🏆 A (90/100) | Desc 100 | Keywords 50 | Δ+41 | 16.7 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (89/100) | Desc 90 | Keywords 84 | Δ+40 | 65.0 tps
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🏆 A (86/100) | Desc 91 | Keywords 81 | Δ+37 | 167.7 tps
- `mlx-community/Qwen3.5-27B-4bit`: 🏆 A (85/100) | Desc 100 | Keywords 51 | Δ+36 | 29.6 tps
- `mlx-community/Ornith-1.0-35B-bf16`: 🏆 A (85/100) | Desc 93 | Keywords 68 | Δ+36 | 60.9 tps

### Watchlist

- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-49 | 181.1 tps | context ignored, harness, long context
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-49 | 179.9 tps | context ignored, harness, long context
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (6/100) | Desc 21 | Keywords 32 | Δ-43 | 24.1 tps | context ignored, harness
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (6/100) | Desc 36 | Keywords 0 | Δ-43 | 60.6 tps | context ignored, degeneration, harness, long context, missing sections
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-29 | 88.4 tps | context ignored, cutoff, degeneration, generation loop, harness, long context, missing sections, repetitive, text sanity

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                        | Verdict   | Hint Handling           | Key Evidence                                               |
|----------------------------------------------|-----------|-------------------------|------------------------------------------------------------|
| `microsoft/Phi-3.5-vision-instruct`          | `clean`   | preserves trusted hints | missing terms: Bird, Boating, Buoy, Bushes, Coast          |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `clean`   | preserves trusted hints | missing terms: Bird, Boating, Buoy, Bushes, Coast          |
| `mlx-community/Ornith-1.0-35B-bf16`          | `clean`   | preserves trusted hints | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary |
| `mlx-community/Qwen3.6-27B-mxfp8`            | `clean`   | preserves trusted hints | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary |

### `caveat`

| Model                                               | Verdict          | Hint Handling                                                             | Key Evidence                                                                                                                                                                                                                                                                                                                     |
|-----------------------------------------------------|------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/pixtral-12b-8bit`                    | `context_budget` | preserves trusted hints \| nonvisual metadata reused                      | output/prompt=2.35% \| nontext prompt burden=87% \| missing terms: Bird, Buoy, Bushes, Coast, Deben Estuary \| nonvisual metadata reused                                                                                                                                                                                         |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `context_budget` | preserves trusted hints \| nonvisual metadata reused                      | output/prompt=3.84% \| nontext prompt burden=86% \| missing terms: Bird, Boating, Bushes, Coast, Forest \| nonvisual metadata reused                                                                                                                                                                                             |
| `mlx-community/GLM-4.6V-Flash-6bit`                 | `context_budget` | preserves trusted hints                                                   | output/prompt=1.41% \| nontext prompt burden=93% \| missing sections: title \| missing terms: Bird, Boating, Bushes, Coast, Deben Estuary                                                                                                                                                                                        |
| `Qwen/Qwen3-VL-2B-Instruct`                         | `context_budget` | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes | Output appears truncated to about 2 tokens. \| At long prompt length (16729 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated). \| output/prompt=0.01% \| nontext prompt burden=97%                                                                                                       |
| `mlx-community/GLM-4.6V-nvfp4`                      | `context_budget` | improves trusted hints                                                    | output/prompt=1.57% \| nontext prompt burden=93% \| missing sections: title \| missing terms: Bird, Boating, Bushes, Coast, Deben Estuary                                                                                                                                                                                        |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`           | `context_budget` | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes | Output appears truncated to about 2 tokens. \| At long prompt length (16729 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated). \| output/prompt=0.01% \| nontext prompt burden=97%                                                                                                       |
| `mlx-community/X-Reasoner-7B-8bit`                  | `context_budget` | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes | Output is very short relative to prompt size (0.1%) with weak text signal 'truncated', suggesting possible early-stop or prompt-handling issues. \| At long prompt length (16740 tokens), output stayed unusually short (14 tokens; ratio 0.1%; weak text signal truncated). \| output/prompt=0.08% \| nontext prompt burden=97% |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `context_budget` | preserves trusted hints \| nonvisual metadata reused                      | output/prompt=0.75% \| nontext prompt burden=97% \| missing terms: Bird, Bushes, Woodbridge, GBR, behind \| keywords=20                                                                                                                                                                                                          |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `context_budget` | preserves trusted hints \| nonvisual metadata reused                      | output/prompt=0.59% \| nontext prompt burden=97% \| missing terms: Bird, Bushes, Coast, Peaceful, Woodbridge \| nonvisual metadata reused                                                                                                                                                                                        |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `context_budget` | preserves trusted hints \| nonvisual metadata reused                      | output/prompt=0.61% \| nontext prompt burden=97% \| missing terms: Bird, Bushes, Coast, Peaceful, Woodbridge \| nonvisual metadata reused                                                                                                                                                                                        |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`           | `context_budget` | preserves trusted hints \| nonvisual metadata reused                      | output/prompt=0.26% \| nontext prompt burden=97% \| missing sections: keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast                                                                                                                                                                                              |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                          | Key Evidence                                                                                                                                                                                 |
|---------------------------------------------------------|---------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-31b-bf16`                        | `semantic_mismatch` | preserves trusted hints \| nonvisual metadata reused                                                   | keywords=57 \| context echo=100% \| nonvisual metadata reused \| reasoning leak                                                                                                              |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing terms: Bird, Buoy, Bushes, Rigging, Woodbridge \| keywords=19 \| nonvisual metadata reused                                                                                           |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing terms: Bird, Boating, Buoy, Bushes, Coast \| nonvisual metadata reused                                                                                                               |
| `qnguyen3/nanoLLaVA`                                    | `model_shortcoming` | degrades trusted hints \| nonvisual metadata reused                                                    | missing sections: keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast \| nonvisual metadata reused                                                                                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing terms: Woodbridge, GBR \| keywords=20 \| nonvisual metadata reused                                                                                                                   |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `harness`           | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes                              | Output appears truncated to about 5 tokens. \| missing terms: Bird, Boat, Boating, Buoy, Bushes                                                                                              |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | `harness`           | degrades trusted hints                                                                                 | Special control token &lt;/think&gt; appeared in generated text. \| missing terms: Bird, Boating, Buoy, Bushes, Coast \| reasoning leak \| text-sanity=gibberish(token_noise)                |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing terms: Bird, Bushes, Forest, Mudflat \| keywords=19 \| nonvisual metadata reused                                                                                                     |
| `mlx-community/FastVLM-0.5B-bf16`                       | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                    | missing sections: keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast \| nonvisual metadata reused                                                                                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing terms: Mast \| keywords=19 \| nonvisual metadata reused                                                                                                                              |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing terms: Mast \| keywords=19 \| nonvisual metadata reused                                                                                                                              |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `semantic_mismatch` | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes                              | missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| text-sanity=numeric_loop                                                               |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Boating, Buoy, Bushes, Coast                                                                                                                                            |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing terms: Bird, Coast, Mudflat, Woodbridge, GBR \| nonvisual metadata reused                                                                                                            |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Boating, Buoy, Bushes, Coast                                                                                                                                            |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `semantic_mismatch` | preserves trusted hints \| nonvisual metadata reused                                                   | missing terms: Mast \| keyword duplication=44% \| context echo=100% \| nonvisual metadata reused                                                                                             |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `model_shortcoming` | ignores trusted hints \| missing terms: Bird, Boating, Buoy, Bushes, Coast                             | missing sections: title, description, keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast                                                                                          |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing terms: Bird, Buoy, Bushes, Coast, Mudflat \| nonvisual metadata reused                                                                                                               |
| `mlx-community/InternVL3-8B-bf16`                       | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Boating, Coast, Deben Estuary, Estuary                                                                                                                                  |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary                                                                                                                                   |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | `semantic_mismatch` | improves trusted hints                                                                                 | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary                                                                                                                                   |
| `mlx-community/InternVL3-14B-8bit`                      | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary                                                                                                                                   |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| nonvisual metadata reused | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| nonvisual metadata reused                                       |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing terms: Bird, Boating, Buoy, Bushes, Coast \| nonvisual metadata reused \| formatting=Unknown tags: &lt;end_of_utterance&gt;                                                          |
| `mlx-community/pixtral-12b-bf16`                        | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary                                                                                                                                   |
| `mlx-community/gemma-4-31b-it-4bit`                     | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary                                                                                                                                   |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `semantic_mismatch` | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes                              | missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| text-sanity=gibberish(token_noise)                                                     |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused                                                    | missing terms: Bird, Bushes, Coast, Forest, Landscape \| keywords=22 \| nonvisual metadata reused                                                                                            |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary                                                                                                                                   |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Buoy, Bushes, Deben Estuary, Forest                                                                                                                                     |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing sections: title \| missing terms: Bird, Boating, Bushes, Mudflat, river \| keyword duplication=39% \| nonvisual metadata reused                                                      |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing sections: title, description, keywords \| missing terms: Bird, Woodbridge, GBR \| nonvisual metadata reused                                                                          |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `cutoff_degraded`   | preserves trusted hints \| nonvisual metadata reused                                                   | hit token cap (500) \| missing sections: title \| missing terms: Bird, Boating, Bushes, Coast, Forest \| nonvisual metadata reused                                                           |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes                              | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| excessive bullets=46                                            |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                    | hit token cap (500) \| missing terms: Bird, Boating, Buoy, Bushes, Coast \| keyword duplication=84% \| nonvisual metadata reused                                                             |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | degrades trusted hints \| nonvisual metadata reused                                                    | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| nonvisual metadata reused                                       |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `model_shortcoming` | degrades trusted hints \| nonvisual metadata reused                                                    | missing sections: title, description, keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast \| nonvisual metadata reused                                                             |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | improves trusted hints \| nonvisual metadata reused                                                    | hit token cap (500) \| missing sections: title, keywords \| missing terms: Boating, Bushes, Deben Estuary, Landscape, Nature \| nonvisual metadata reused                                    |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `model_shortcoming` | degrades trusted hints \| nonvisual metadata reused                                                    | missing sections: title, description, keywords \| missing terms: Bird, Boating, Buoy, Bushes, Foliage \| nonvisual metadata reused                                                           |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `model_shortcoming` | degrades trusted hints \| nonvisual metadata reused                                                    | missing sections: title, description, keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast \| nonvisual metadata reused                                                             |
| `mlx-community/MolmoPoint-8B-fp16`                      | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Deben Estuary, Mudflat, Peaceful, Deben                                                                                                                                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused                                                   | missing sections: title \| missing terms: Bird, Boating, Bushes, Deben Estuary, Estuary \| nonvisual metadata reused \| reasoning leak                                                       |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes                              | At long prompt length (16731 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary                                                                                                                                   |
| `mlx-community/Qwen3.5-27B-4bit`                        | `semantic_mismatch` | preserves trusted hints                                                                                | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary                                                                                                                                   |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `semantic_mismatch` | improves trusted hints                                                                                 | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary                                                                                                                                   |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                                | Evidence Snapshot                                                                                                                                                                                      | Affected Models                                     | Issue Draft                                                                                                                              | Evidence Bundle                                                                                                                                                                                           | Fixed When                                                   |
|----------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| `mlx-vlm`                                                | mlx-vlm: Decode / model error: list index out of range | Model Error \| phase decode \| IndexError                                                                                                                                                              | 1: `mlx-community/gemma-4-31b-bf16`                 | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx-vlm_mlx-vlm-decode-model_001.md)         | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_007_mlx-community_gemma-4-31b-bf16_MLX_VLM_DECODE_MODEL_eddb255ac4b7.json)                       | Load/generation completes or fails with a narrower owner.    |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text         | decoded text contains control token &lt;/think&gt; \| prompt=1,127 \| output/prompt=6.83% \| nontext burden=61% \| stop=completed                                                                      | 1: `mlx-community/MiniCPM-V-4.6-8bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_stop-token_001.md)                   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_002_mlx-community_MiniCPM-V-4.6-8bit_mlx_vlm_stop_token_001.json)                                | No leaked stop/control tokens.                               |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                  | generated_tokens~5 \| prompt=1,530 \| output/prompt=0.33% \| nontext burden=72% \| stop=completed                                                                                                      | 1: `mlx-community/paligemma2-3b-ft-docci-448-bf16`  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_model-config-mlx-vlm_prompt-template_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_010_mlx-community_paligemma2-3b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json) | Requested sections render without template leakage.          |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short  | generated_tokens~2 \| prompt_tokens=16729, output_tokens=2, output/prompt=0.0%, weak text=truncated \| prompt=16,729 \| output/prompt=0.01% \| nontext burden=97% \| stop=completed \| 3 model cluster | 3: `Qwen/Qwen3-VL-2B-Instruct` (+2)                 | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm-mlx_long-context_001.md)             | [3 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_001_Qwen_Qwen3-VL-2B-Instruct_mlx_vlm_mlx_long_context_001.json)                              | Full and reduced reruns avoid context collapse.              |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short  | prompt_tokens=16731, repetitive output \| prompt=16,731 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500)                                                         | 1: `mlx-community/Qwen3-VL-2B-Thinking-bf16`        | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm-mlx_long-context_002.md)             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_004_mlx-community_Qwen3-VL-2B-Thinking-bf16_mlx_vlm_mlx_long_context_002.json)                   | Full and reduced reruns avoid context collapse.              |
| model repository                                         | Generated text is mixed-script token-soup              | text sanity \| gibberish(token noise) \| low hint overlap \| prompt=1,530 \| output/prompt=1.63% \| nontext burden=72% \| stop=completed                                                               | 1: `mlx-community/paligemma2-10b-ft-docci-448-bf16` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_model_text-sanity_001.md)                    | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_009_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_text_sanity_001.json)                   | Generated text is readable natural language, not token soup. |
| model repository                                         | Generated text is mixed-script token-soup              | text sanity \| numeric loop \| trusted overlap \| prompt=622 \| output/prompt=39.87% \| nontext burden=30% \| stop=completed \| 2 model cluster                                                        | 2: `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (+1)  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_007_model_text-sanity_002.md)                    | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_005_mlx-community_SmolVLM2-2.2B-Instruct-mlx_model_text_sanity_002.json)                      | Generated text is readable natural language, not token soup. |
<!-- markdownlint-enable MD060 -->

## Model Verdicts

### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_DECODE_MODEL`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=57; context echo=100%; nonvisual metadata reused;
  reasoning leak
- _Tokens:_ prompt n/a; estimated text 435 tok; estimated non-text n/a;
  generated n/a; requested max 500 tok; stop reason exception


### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Buoy, Bushes, Rigging, Woodbridge;
  keywords=19; nonvisual metadata reused
- _Tokens:_ prompt 585 tok; estimated text 435 tok; estimated non-text 150
  tok; generated 121 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast; nonvisual
  metadata reused
- _Tokens:_ prompt 775 tok; estimated text 435 tok; estimated non-text 340
  tok; generated 69 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bird, Boating,
  Buoy, Bushes, Coast; nonvisual metadata reused
- _Tokens:_ prompt 512 tok; estimated text 435 tok; estimated non-text 77 tok;
  generated 31 tok; requested max 500 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Woodbridge, GBR; keywords=20; nonvisual
  metadata reused
- _Tokens:_ prompt 512 tok; estimated text 435 tok; estimated non-text 77 tok;
  generated 93 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 5 tokens.; missing terms:
  Bird, Boat, Boating, Buoy, Bushes
- _Tokens:_ prompt 1530 tok; estimated text 435 tok; estimated non-text 1095
  tok; generated 5 tok; requested max 500 tok; stop reason completed


### `mlx-community/MiniCPM-V-4.6-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; missing terms: Bird, Boating, Buoy, Bushes, Coast; reasoning leak;
  text-sanity=gibberish(token_noise)
- _Tokens:_ prompt 1127 tok; estimated text 435 tok; estimated non-text 692
  tok; generated 77 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Forest, Mudflat; keywords=19;
  nonvisual metadata reused
- _Tokens:_ prompt 775 tok; estimated text 435 tok; estimated non-text 340
  tok; generated 171 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bird, Boating,
  Buoy, Bushes, Coast; nonvisual metadata reused
- _Tokens:_ prompt 516 tok; estimated text 435 tok; estimated non-text 81 tok;
  generated 191 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Mast; keywords=19; nonvisual metadata reused
- _Tokens:_ prompt 1722 tok; estimated text 435 tok; estimated non-text 1287
  tok; generated 90 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Mast; keywords=19; nonvisual metadata reused
- _Tokens:_ prompt 1722 tok; estimated text 435 tok; estimated non-text 1287
  tok; generated 90 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; text-sanity=numeric_loop
- _Tokens:_ prompt 1530 tok; estimated text 435 tok; estimated non-text 1095
  tok; generated 27 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Tokens:_ prompt 783 tok; estimated text 435 tok; estimated non-text 348
  tok; generated 88 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Tokens:_ prompt 1330 tok; estimated text 435 tok; estimated non-text 895
  tok; generated 101 tok; requested max 500 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Coast, Mudflat, Woodbridge, GBR;
  nonvisual metadata reused
- _Tokens:_ prompt 783 tok; estimated text 435 tok; estimated non-text 348
  tok; generated 89 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Tokens:_ prompt 3123 tok; estimated text 435 tok; estimated non-text 2688
  tok; generated 107 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Tokens:_ prompt 1330 tok; estimated text 435 tok; estimated non-text 895
  tok; generated 101 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Mast; keyword duplication=44%; context
  echo=100%; nonvisual metadata reused
- _Tokens:_ prompt 622 tok; estimated text 435 tok; estimated non-text 187
  tok; generated 248 tok; requested max 500 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boating, Buoy, Bushes, Coast
- _Tokens:_ prompt 2730 tok; estimated text 435 tok; estimated non-text 2295
  tok; generated 11 tok; requested max 500 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Buoy, Bushes, Coast, Mudflat; nonvisual
  metadata reused
- _Tokens:_ prompt 783 tok; estimated text 435 tok; estimated non-text 348
  tok; generated 78 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Coast, Deben Estuary, Estuary
- _Tokens:_ prompt 2295 tok; estimated text 435 tok; estimated non-text 1860
  tok; generated 71 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary
- _Tokens:_ prompt 778 tok; estimated text 435 tok; estimated non-text 343
  tok; generated 78 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Tokens:_ prompt 3124 tok; estimated text 435 tok; estimated non-text 2689
  tok; generated 95 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 87% and the output stays weak under that load.
- _Key signals:_ output/prompt=2.35%; nontext prompt burden=87%; missing
  terms: Bird, Buoy, Bushes, Coast, Deben Estuary; nonvisual metadata reused
- _Tokens:_ prompt 3317 tok; estimated text 435 tok; estimated non-text 2882
  tok; generated 78 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary
- _Tokens:_ prompt 2295 tok; estimated text 435 tok; estimated non-text 1860
  tok; generated 72 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ output/prompt=3.84%; nontext prompt burden=86%; missing
  terms: Bird, Boating, Bushes, Coast, Forest; nonvisual metadata reused
- _Tokens:_ prompt 3124 tok; estimated text 435 tok; estimated non-text 2689
  tok; generated 120 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes; nonvisual
  metadata reused
- _Tokens:_ prompt 769 tok; estimated text 435 tok; estimated non-text 334
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast; nonvisual
  metadata reused; formatting=Unknown tags: &lt;end_of_utterance&gt;
- _Tokens:_ prompt 2805 tok; estimated text 435 tok; estimated non-text 2370
  tok; generated 126 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary
- _Tokens:_ prompt 3317 tok; estimated text 435 tok; estimated non-text 2882
  tok; generated 80 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Tokens:_ prompt 783 tok; estimated text 435 tok; estimated non-text 348
  tok; generated 81 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; text-sanity=gibberish(token_noise)
- _Tokens:_ prompt 1530 tok; estimated text 435 tok; estimated non-text 1095
  tok; generated 25 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Forest, Landscape;
  keywords=22; nonvisual metadata reused
- _Tokens:_ prompt 777 tok; estimated text 435 tok; estimated non-text 342
  tok; generated 272 tok; requested max 500 tok; stop reason completed


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Tokens:_ prompt 2591 tok; estimated text 435 tok; estimated non-text 2156
  tok; generated 82 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Buoy, Bushes, Deben Estuary, Forest
- _Tokens:_ prompt 778 tok; estimated text 435 tok; estimated non-text 343
  tok; generated 91 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Bird, Boating,
  Bushes, Mudflat, river; keyword duplication=39%; nonvisual metadata reused
- _Tokens:_ prompt 1514 tok; estimated text 435 tok; estimated non-text 1079
  tok; generated 393 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 93% and the output stays weak under that load.
- _Key signals:_ output/prompt=1.41%; nontext prompt burden=93%; missing
  sections: title; missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Tokens:_ prompt 6542 tok; estimated text 435 tok; estimated non-text 6107
  tok; generated 92 tok; requested max 500 tok; stop reason completed


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Woodbridge, GBR; nonvisual metadata reused
- _Tokens:_ prompt 493 tok; estimated text 435 tok; estimated non-text 58 tok;
  generated 129 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  Bird, Boating, Bushes, Coast, Forest; nonvisual metadata reused
- _Tokens:_ prompt 1514 tok; estimated text 435 tok; estimated non-text 1079
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes; excessive
  bullets=46
- _Tokens:_ prompt 4602 tok; estimated text 435 tok; estimated non-text 4167
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: Bird, Boating, Buoy,
  Bushes, Coast; keyword duplication=84%; nonvisual metadata reused
- _Tokens:_ prompt 6542 tok; estimated text 435 tok; estimated non-text 6107
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes; nonvisual
  metadata reused
- _Tokens:_ prompt 1826 tok; estimated text 435 tok; estimated non-text 1391
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `Qwen/Qwen3-VL-2B-Instruct`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 2 tokens.; At long prompt
  length (16729 tokens), output stayed unusually short (2 tokens; ratio 0.0%;
  weak text signal truncated).; output/prompt=0.01%; nontext prompt burden=97%
- _Tokens:_ prompt 16729 tok; estimated text 435 tok; estimated non-text 16294
  tok; generated 2 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boating, Buoy, Bushes, Coast; nonvisual metadata reused
- _Tokens:_ prompt 1699 tok; estimated text 435 tok; estimated non-text 1264
  tok; generated 31 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 93% and the output stays weak under that load.
- _Key signals:_ output/prompt=1.57%; nontext prompt burden=93%; missing
  sections: title; missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Tokens:_ prompt 6542 tok; estimated text 435 tok; estimated non-text 6107
  tok; generated 103 tok; requested max 500 tok; stop reason completed


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, keywords;
  missing terms: Boating, Bushes, Deben Estuary, Landscape, Nature; nonvisual
  metadata reused
- _Tokens:_ prompt 3408 tok; estimated text 435 tok; estimated non-text 2973
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boating, Buoy, Bushes, Foliage; nonvisual metadata reused
- _Tokens:_ prompt 1699 tok; estimated text 435 tok; estimated non-text 1264
  tok; generated 42 tok; requested max 500 tok; stop reason completed


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boating, Buoy, Bushes, Coast; nonvisual metadata reused
- _Tokens:_ prompt 494 tok; estimated text 435 tok; estimated non-text 59 tok;
  generated 73 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 2 tokens.; At long prompt
  length (16729 tokens), output stayed unusually short (2 tokens; ratio 0.0%;
  weak text signal truncated).; output/prompt=0.01%; nontext prompt burden=97%
- _Tokens:_ prompt 16729 tok; estimated text 435 tok; estimated non-text 16294
  tok; generated 2 tok; requested max 500 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ Output is very short relative to prompt size (0.1%) with weak
  text signal 'truncated', suggesting possible early-stop or prompt-handling
  issues.; At long prompt length (16740 tokens), output stayed unusually short
  (14 tokens; ratio 0.1%; weak text signal truncated).; output/prompt=0.08%;
  nontext prompt burden=97%
- _Tokens:_ prompt 16740 tok; estimated text 435 tok; estimated non-text 16305
  tok; generated 14 tok; requested max 500 tok; stop reason completed


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Deben Estuary, Mudflat, Peaceful, Deben
- _Tokens:_ prompt 3327 tok; estimated text 435 tok; estimated non-text 2892
  tok; generated 94 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Bird, Boating,
  Bushes, Deben Estuary, Estuary; nonvisual metadata reused; reasoning leak
- _Tokens:_ prompt 1514 tok; estimated text 435 tok; estimated non-text 1079
  tok; generated 96 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16731 tokens), output became
  repetitive.; hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes
- _Tokens:_ prompt 16731 tok; estimated text 435 tok; estimated non-text 16296
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.75%; nontext prompt burden=97%; missing
  terms: Bird, Bushes, Woodbridge, GBR, behind; keywords=20
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 126 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 106 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.59%; nontext prompt burden=97%; missing
  terms: Bird, Bushes, Coast, Peaceful, Woodbridge; nonvisual metadata reused
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 99 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.61%; nontext prompt burden=97%; missing
  terms: Bird, Bushes, Coast, Peaceful, Woodbridge; nonvisual metadata reused
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 102 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ornith-1.0-35B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 105 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.26%; nontext prompt burden=97%; missing
  sections: keywords; missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Tokens:_ prompt 16740 tok; estimated text 435 tok; estimated non-text 16305
  tok; generated 44 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 114 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 130 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 135 tok; requested max 500 tok; stop reason completed

