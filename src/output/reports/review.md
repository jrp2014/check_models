<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

Generated on: 2026-07-17 13:44:05 BST

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Companion artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Qwen3.6-27B-mxfp8`: 🏆 A (85/100) | Desc 93 | Keywords 68 | Δ+36 | 6.3 tps
- `mlx-community/gemma-3-27b-it-qat-4bit`: ✅ B (79/100) | Desc 92 | Keywords 80 | Δ+30 | 18.9 tps
- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ✅ B (76/100) | Desc 90 | Keywords 77 | Δ+27 | 503.6 tps
- `mlx-community/gemma-4-31b-it-4bit`: ✅ B (74/100) | Desc 93 | Keywords 68 | Δ+25 | 19.6 tps
- `mlx-community/gemma-3-27b-it-qat-8bit`: ✅ B (72/100) | Desc 93 | Keywords 66 | Δ+23 | 17.7 tps

### Watchlist

- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (4/100) | Desc 45 | Keywords 0 | Δ-45 | 46.4 tps | context ignored, harness, long context, missing sections
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-29 | 91.9 tps | context ignored, cutoff, degeneration, generation loop, harness, long context, missing sections, repetitive, text sanity
- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-29 | 92.5 tps | context ignored, cutoff, degeneration, generation loop, harness, long context, missing sections, repetitive, text sanity
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-29 | 27.9 tps | context ignored, cutoff, degeneration, generation loop, harness, long context, missing sections, repetitive, text sanity
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (50/100) | Desc 60 | Keywords 50 | Δ+0 | 207.3 tps | cutoff, generation loop, harness, long context, missing sections, repetitive, trusted hint degraded

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                          | Verdict   | Hint Handling                                    | Key Evidence                                                                       |
|------------------------------------------------|-----------|--------------------------------------------------|------------------------------------------------------------------------------------|
| `mlx-community/diffusiongemma-26B-A4B-it-8bit` | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Bird, Coast, Mudflat, Peaceful, Woodbridge \| low-draft-improvement |
| `mlx-community/gemma-4-31b-it-4bit`            | `clean`   | preserves trusted hints \| low-draft-improvement | missing terms: Bird, Boating, Bushes, Coast, Mudflat \| low-draft-improvement      |
| `mlx-community/Qwen3.6-27B-mxfp8`              | `clean`   | improves trusted hints                           | missing terms: Bird, Boating, Bushes, Coast, Forest                                |

### `caveat`

| Model                                              | Verdict          | Hint Handling                                                                                      | Key Evidence                                                                                                                                                                                                                                                                                                      |
|----------------------------------------------------|------------------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                 | `clean`          | improves trusted hints                                                                             | missing terms: Bird, Buoy, Mudflat, behind, bank                                                                                                                                                                                                                                                                  |
| `mlx-community/nanoLLaVA-1.5-4bit`                 | `clean`          | preserves trusted hints \| low-draft-improvement                                                   | keywords=21 \| low-draft-improvement                                                                                                                                                                                                                                                                              |
| `mlx-community/Idefics3-8B-Llama3-bf16`            | `clean`          | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Bird, Boating, Buoy, Bushes, Coast \| formatting=Unknown tags: &lt;end_of_utterance&gt; \| low-draft-improvement                                                                                                                                                                                   |
| `mlx-community/gemma-3-27b-it-qat-8bit`            | `clean`          | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Bird, Bushes, Coast, Mudflat, Peaceful \| keywords=19 \| low-draft-improvement                                                                                                                                                                                                                     |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `clean`          | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Bird, Deben, Woodbridge, GBR, bank \| keywords=19 \| low-draft-improvement                                                                                                                                                                                                                         |
| `mlx-community/gemma-3-27b-it-qat-4bit`            | `clean`          | improves trusted hints \| low-draft-improvement                                                    | missing terms: Bird, Bushes, Forest, Peaceful, Rigging \| keywords=21 \| low-draft-improvement                                                                                                                                                                                                                    |
| `mlx-community/llava-v1.6-mistral-7b-8bit`         | `token_cap`      | preserves trusted hints \| low-draft-improvement                                                   | hit token cap (500) \| keywords=27 \| low-draft-improvement                                                                                                                                                                                                                                                       |
| `mlx-community/GLM-4.6V-Flash-6bit`                | `context_budget` | improves trusted hints                                                                             | output/prompt=1.46% \| visual input burden=93% \| missing sections: title \| missing terms: Bird, Boating, Buoy, Bushes, Coast                                                                                                                                                                                    |
| `mlx-community/GLM-4.6V-nvfp4`                     | `context_budget` | preserves trusted hints \| low-draft-improvement                                                   | output/prompt=1.70% \| visual input burden=93% \| missing sections: title \| missing terms: Bird, Coast, Deben, Woodbridge, GBR                                                                                                                                                                                   |
| `mlx-community/Molmo-7B-D-0924-bf16`               | `clean`          | preserves trusted hints \| low-draft-improvement                                                   | missing terms: GBR, bank \| keywords=20 \| low-draft-improvement                                                                                                                                                                                                                                                  |
| `mlx-community/MolmoPoint-8B-fp16`                 | `clean`          | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Bird, Deben, Woodbridge, GBR, sailing \| keywords=19 \| low-draft-improvement                                                                                                                                                                                                                      |
| `mlx-community/Molmo-7B-D-0924-8bit`               | `clean`          | preserves trusted hints \| low-draft-improvement                                                   | missing terms: GBR, bank \| keywords=20 \| low-draft-improvement                                                                                                                                                                                                                                                  |
| `mlx-community/X-Reasoner-7B-8bit`                 | `context_budget` | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement | Output is very short relative to prompt size (0.1%) with weak text signal 'truncated', suggesting possible early-stop or prompt-handling issues. \| At mixed burden (16790 tokens), output stayed unusually short (14 tokens; ratio 0.1%; weak text signal truncated). \| output/prompt=0.08% \| mixed burden=97% |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling                                                                                      | Key Evidence                                                                                                                                                                           |
|---------------------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-31b-bf16`                        | `runtime_failure`   | not evaluated                                                                                      | model error \| mlx vlm decode model                                                                                                                                                    |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `model_shortcoming` | preserves trusted hints \| unverified-context-copy \| low-draft-improvement                        | keywords=20 \| unverified-context-copy \| low-draft-improvement                                                                                                                        |
| `qnguyen3/nanoLLaVA`                                    | `model_shortcoming` | degrades trusted hints \| low-draft-improvement                                                    | missing sections: keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast \| low-draft-improvement                                                                               |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `model_shortcoming` | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement | missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement                                                            |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | `harness`           | preserves trusted hints                                                                            | Special control token &lt;/think&gt; appeared in generated text. \| missing terms: Bird, Boating, Bushes, Coast, Estuary \| reasoning leak \| text-sanity=gibberish(token_noise)       |
| `mlx-community/FastVLM-0.5B-bf16`                       | `model_shortcoming` | preserves trusted hints \| unverified-context-copy \| low-draft-improvement                        | keywords=28 \| unverified-context-copy \| low-draft-improvement                                                                                                                        |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `model_shortcoming` | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement | missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement                                                            |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | `semantic_mismatch` | preserves trusted hints                                                                            | missing terms: Bird, Boating, Buoy, Bushes, Coast                                                                                                                                      |
| `microsoft/Phi-3.5-vision-instruct`                     | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Boating, Buoy, Bushes, Coast, Foliage \| keywords=21 \| low-draft-improvement                                                                                           |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                   | hit token cap (500) \| missing sections: description, keywords \| missing terms: Bird, Bushes, Forest, Mast, Mudflat \| repetitive token=phrase: "boating, boats, moored, buoy,..."    |
| `mlx-community/InternVL3-8B-bf16`                       | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Bird, Landscape, Mudflat, Peaceful, Rigging \| low-draft-improvement                                                                                                    |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | `semantic_mismatch` | improves trusted hints                                                                             | missing terms: Bird, Boating, Bushes, Coast, Estuary                                                                                                                                   |
| `mlx-community/pixtral-12b-8bit`                        | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Bird, Boating, Bushes, Coast, Forest \| low-draft-improvement                                                                                                           |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `model_shortcoming` | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement | missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement                                                            |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | `semantic_mismatch` | improves trusted hints                                                                             | missing terms: Bird, Boating, Bushes, Coast, Landscape                                                                                                                                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `model_shortcoming` | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement | missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement                                                            |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `model_shortcoming` | preserves trusted hints \| unverified-context-copy \| low-draft-improvement                        | missing terms: Mast \| keyword duplication=41% \| unverified-context-copy \| low-draft-improvement                                                                                     |
| `mlx-community/pixtral-12b-bf16`                        | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Bird, Boating, Bushes, Coast, Foliage \| low-draft-improvement                                                                                                          |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | `semantic_mismatch` | improves trusted hints                                                                             | missing terms: Bird, Boating, Bushes, Coast, Forest                                                                                                                                    |
| `mlx-community/InternVL3-14B-8bit`                      | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Bird, Boating, Bushes, Coast, Mudflat \| low-draft-improvement                                                                                                          |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Bird, Boating, Buoy, Bushes, Coast \| low-draft-improvement                                                                                                             |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes                          | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| excessive bullets=50                                      |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Bird, Bushes, Coast, Landscape, Mudflat \| low-draft-improvement                                                                                                        |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Boating, Buoy, Bushes, Coast, Foliage \| keywords=21 \| low-draft-improvement                                                                                           |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `model_shortcoming` | improves trusted hints \| unverified-context-copy                                                  | missing terms: Bird, Buoy, Bushes, Coast, Foliage \| keywords=27 \| unverified-context-copy                                                                                            |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                   | hit token cap (500) \| missing sections: title \| missing terms: GBR \| keywords=59                                                                                                    |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                   | hit token cap (500) \| missing sections: title, description \| missing terms: Bird, Boating, Coast, Landscape, Mudflat \| keywords=24                                                  |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                   | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bird, Boating, Bushes, Coast, Foliage \| low-draft-improvement                                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes                          | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| excessive bullets=46                                      |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | improves trusted hints                                                                             | hit token cap (500) \| missing terms: Bird, Boating, Bushes, Coast, Forest \| keyword duplication=80% \| repetitive token=phrase: "black hull, white superstructu..."                  |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | improves trusted hints                                                                             | hit token cap (500) \| missing sections: title, keywords \| missing terms: Bird, Boating, Bushes, Coast, Forest \| reasoning leak                                                      |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement | At mixed burden (16779 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes |
| `Qwen/Qwen3-VL-2B-Instruct`                             | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement | At mixed burden (16779 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes                          | hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| excessive bullets=34                                      |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `model_shortcoming` | degrades trusted hints \| low-draft-improvement                                                    | missing sections: title, description, keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast \| low-draft-improvement                                                           |
| `mlx-community/Ornith-1.0-35B-bf16`                     | `semantic_mismatch` | improves trusted hints                                                                             | missing terms: Bird, Boating, Bushes, Coast, Mudflat                                                                                                                                   |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `semantic_mismatch` | improves trusted hints                                                                             | missing terms: Bird, Boat, Boating, Bushes, Coast                                                                                                                                      |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | `cutoff_degraded`   | ignores trusted hints \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement | At mixed burden (16781 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `semantic_mismatch` | preserves trusted hints                                                                            | missing terms: Bird, Boating, Bushes, Coast, Estuary                                                                                                                                   |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `semantic_mismatch` | preserves trusted hints \| low-draft-improvement                                                   | missing terms: Bird, Boating, Bushes, Coast, Forest \| low-draft-improvement                                                                                                           |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `semantic_mismatch` | preserves trusted hints                                                                            | missing terms: Bird, Boating, Bushes, Coast, Landscape                                                                                                                                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `cutoff_degraded`   | degrades trusted hints \| low-draft-improvement                                                    | At mixed burden (16790 tokens), output became repetitive. \| hit token cap (500) \| missing sections: description, keywords \| missing terms: Bird, Buoy, Bushes, Coast, Estuary       |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `semantic_mismatch` | improves trusted hints                                                                             | missing terms: Bird, Boating, Bushes, Coast, Foliage                                                                                                                                   |
| `mlx-community/Qwen3.5-27B-4bit`                        | `semantic_mismatch` | improves trusted hints                                                                             | missing terms: Bird, Boating, Bushes, Coast, Mudflat                                                                                                                                   |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `cutoff_degraded`   | preserves trusted hints \| low-draft-improvement                                                   | hit token cap (500) \| missing sections: title \| missing terms: Mudflat, Peaceful, Riverbank, Woodbridge, GBR \| keyword duplication=50%                                              |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

<!-- markdownlint-disable MD060 -->

| Target                                         | Problem                                                | Evidence Snapshot                                                                                                                                                                                       | Affected Models                                    | Issue Draft                                                                                                                                     | Evidence Bundle                                                                                                                                                                        | Fixed When                                                   |
|------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| mlx-vlm first; MLX if cache/runtime reproduces | mlx-vlm: Decode / model error: list index out of range | Model Error \| phase decode \| IndexError                                                                                                                                                               | 1: `mlx-community/gemma-4-31b-bf16`                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_unresolved-mlx-mlx-vlm_mlx-vlm-decode-model_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T122048Z_001_mlx-community_gemma-4-31b-bf16_MLX_VLM_DECODE_MODEL_0375357f22c1.json)    | Load/generation completes or fails with a narrower owner.    |
| `mlx-vlm`                                      | Stop/control tokens leaked into generated text         | decoded text contains control token &lt;/think&gt; \| prompt=1,179 \| output/prompt=6.87% \| stop=completed                                                                                             | 1: `mlx-community/MiniCPM-V-4.6-8bit`              | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_stop-token_001.md)                          | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T122048Z_002_mlx-community_MiniCPM-V-4.6-8bit_mlx_vlm_stop_token_001.json)             | No leaked stop/control tokens.                               |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short  | prompt_tokens=16779, repetitive output \| prompt=16,779 \| output/prompt=2.98% \| mixed burden=97% \| stop=max_tokens \| hit token cap (500) \| 4 model cluster                                         | 4: `Qwen/Qwen3-VL-2B-Instruct` (+3)                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm-mlx_long-context_001.md)                    | [4 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T122048Z_005_Qwen_Qwen3-VL-2B-Instruct_mlx_vlm_mlx_long_context_001.json)           | Full and reduced reruns avoid context collapse.              |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short  | output/prompt=0.1%, weak text=truncated \| prompt_tokens=16790, output_tokens=14, output/prompt=0.1%, weak text=truncated \| prompt=16,790 \| output/prompt=0.08% \| mixed burden=97% \| stop=completed | 1: `mlx-community/X-Reasoner-7B-8bit`              | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm-mlx_long-context_002.md)                    | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T122048Z_006_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_002.json)       | Full and reduced reruns avoid context collapse.              |
| model repository                               | Generated text is mixed-script token-soup              | token cap \| missing sections \| abrupt tail \| prompt=3,457 \| output/prompt=14.46% \| mixed burden=86% \| stop=max_tokens \| hit token cap (500)                                                      | 1: `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_model_text-sanity_001.md)                           | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T122048Z_003_mlx-community_Apriel-1.5-15b-Thinker-6bit-MLX_model_text_sanity_001.json) | Generated text is readable natural language, not token soup. |
<!-- markdownlint-enable MD060 -->

## Model Verdicts

### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `unresolved: mlx/mlx-vlm`; reported package `mlx-vlm`;
  failure stage `Model Error`; diagnostic code `MLX_VLM_DECODE_MODEL`
- _Next step:_ Inspect the canonical log and diagnostics output.
- _Key signals:_ model error; mlx vlm decode model
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Buoy, Mudflat, behind, bank
- _Tokens:_ prompt 639 tok; estimated text 478 tok; estimated non-text 161
  tok; generated 181 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; unverified-context-copy; low-draft-improvement
- _Tokens:_ prompt 829 tok; estimated text 478 tok; estimated non-text 351
  tok; generated 144 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bird, Boating,
  Buoy, Bushes, Coast; low-draft-improvement
- _Tokens:_ prompt 562 tok; estimated text 478 tok; estimated non-text 84 tok;
  generated 37 tok; requested max 500 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=21; low-draft-improvement
- _Tokens:_ prompt 562 tok; estimated text 478 tok; estimated non-text 84 tok;
  generated 98 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; low-draft-improvement
- _Tokens:_ prompt 1776 tok; estimated text 478 tok; estimated non-text 1298
  tok; generated 18 tok; requested max 500 tok; stop reason completed


### `mlx-community/MiniCPM-V-4.6-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; missing terms: Bird, Boating, Bushes, Coast, Estuary; reasoning leak;
  text-sanity=gibberish(token_noise)
- _Tokens:_ prompt 1179 tok; estimated text 478 tok; estimated non-text 701
  tok; generated 81 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=28; unverified-context-copy; low-draft-improvement
- _Tokens:_ prompt 566 tok; estimated text 478 tok; estimated non-text 88 tok;
  generated 171 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; low-draft-improvement
- _Tokens:_ prompt 1582 tok; estimated text 478 tok; estimated non-text 1104
  tok; generated 17 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Tokens:_ prompt 835 tok; estimated text 478 tok; estimated non-text 357
  tok; generated 86 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Boating, Buoy, Bushes, Coast, Foliage;
  keywords=21; low-draft-improvement
- _Tokens:_ prompt 1394 tok; estimated text 478 tok; estimated non-text 916
  tok; generated 130 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: description, keywords;
  missing terms: Bird, Bushes, Forest, Mast, Mudflat; repetitive token=phrase:
  "boating, boats, moored, buoy,..."
- _Tokens:_ prompt 829 tok; estimated text 478 tok; estimated non-text 351
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Landscape, Mudflat, Peaceful, Rigging;
  low-draft-improvement
- _Tokens:_ prompt 2345 tok; estimated text 478 tok; estimated non-text 1867
  tok; generated 81 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Estuary
- _Tokens:_ prompt 3173 tok; estimated text 478 tok; estimated non-text 2695
  tok; generated 105 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Forest;
  low-draft-improvement
- _Tokens:_ prompt 3366 tok; estimated text 478 tok; estimated non-text 2888
  tok; generated 92 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; low-draft-improvement
- _Tokens:_ prompt 1582 tok; estimated text 478 tok; estimated non-text 1104
  tok; generated 17 tok; requested max 500 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast;
  formatting=Unknown tags: &lt;end_of_utterance&gt;; low-draft-improvement
- _Tokens:_ prompt 2855 tok; estimated text 478 tok; estimated non-text 2377
  tok; generated 109 tok; requested max 500 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Coast, Mudflat, Peaceful, Woodbridge;
  low-draft-improvement
- _Tokens:_ prompt 835 tok; estimated text 478 tok; estimated non-text 357
  tok; generated 89 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Landscape
- _Tokens:_ prompt 3172 tok; estimated text 478 tok; estimated non-text 2694
  tok; generated 125 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; low-draft-improvement
- _Tokens:_ prompt 1776 tok; estimated text 478 tok; estimated non-text 1298
  tok; generated 18 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Mast; keyword duplication=41%;
  unverified-context-copy; low-draft-improvement
- _Tokens:_ prompt 676 tok; estimated text 478 tok; estimated non-text 198
  tok; generated 275 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Foliage;
  low-draft-improvement
- _Tokens:_ prompt 3366 tok; estimated text 478 tok; estimated non-text 2888
  tok; generated 88 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Forest
- _Tokens:_ prompt 3173 tok; estimated text 478 tok; estimated non-text 2695
  tok; generated 108 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Mudflat;
  low-draft-improvement
- _Tokens:_ prompt 835 tok; estimated text 478 tok; estimated non-text 357
  tok; generated 90 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Mudflat;
  low-draft-improvement
- _Tokens:_ prompt 2345 tok; estimated text 478 tok; estimated non-text 1867
  tok; generated 109 tok; requested max 500 tok; stop reason completed


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast;
  low-draft-improvement
- _Tokens:_ prompt 2640 tok; estimated text 478 tok; estimated non-text 2162
  tok; generated 96 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes; excessive
  bullets=50
- _Tokens:_ prompt 821 tok; estimated text 478 tok; estimated non-text 343
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Landscape, Mudflat;
  low-draft-improvement
- _Tokens:_ prompt 835 tok; estimated text 478 tok; estimated non-text 357
  tok; generated 96 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Boating, Buoy, Bushes, Coast, Foliage;
  keywords=21; low-draft-improvement
- _Tokens:_ prompt 1394 tok; estimated text 478 tok; estimated non-text 916
  tok; generated 130 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Buoy, Bushes, Coast, Foliage;
  keywords=27; unverified-context-copy
- _Tokens:_ prompt 829 tok; estimated text 478 tok; estimated non-text 351
  tok; generated 332 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Mudflat, Peaceful;
  keywords=19; low-draft-improvement
- _Tokens:_ prompt 830 tok; estimated text 478 tok; estimated non-text 352
  tok; generated 108 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  GBR; keywords=59
- _Tokens:_ prompt 1564 tok; estimated text 478 tok; estimated non-text 1086
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description;
  missing terms: Bird, Boating, Coast, Landscape, Mudflat; keywords=24
- _Tokens:_ prompt 1564 tok; estimated text 478 tok; estimated non-text 1086
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Deben, Woodbridge, GBR, bank;
  keywords=19; low-draft-improvement
- _Tokens:_ prompt 543 tok; estimated text 478 tok; estimated non-text 65 tok;
  generated 148 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Forest, Peaceful, Rigging;
  keywords=21; low-draft-improvement
- _Tokens:_ prompt 830 tok; estimated text 478 tok; estimated non-text 352
  tok; generated 105 tok; requested max 500 tok; stop reason completed


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boating, Bushes, Coast, Foliage;
  low-draft-improvement
- _Tokens:_ prompt 1883 tok; estimated text 478 tok; estimated non-text 1405
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); keywords=27; low-draft-improvement
- _Tokens:_ prompt 2791 tok; estimated text 478 tok; estimated non-text 2313
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes; excessive
  bullets=46
- _Tokens:_ prompt 4654 tok; estimated text 478 tok; estimated non-text 4176
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: Bird, Boating, Bushes,
  Coast, Forest; keyword duplication=80%; repetitive token=phrase: "black
  hull, white superstructu..."
- _Tokens:_ prompt 6592 tok; estimated text 478 tok; estimated non-text 6114
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a visual input burden issue first; reduce that
  input load or inspect long-context handling before judging output quality.
- _Key signals:_ output/prompt=1.46%; visual input burden=93%; missing
  sections: title; missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Tokens:_ prompt 6592 tok; estimated text 478 tok; estimated non-text 6114
  tok; generated 96 tok; requested max 500 tok; stop reason completed


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, keywords;
  missing terms: Bird, Boating, Bushes, Coast, Forest; reasoning leak
- _Tokens:_ prompt 3457 tok; estimated text 478 tok; estimated non-text 2979
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a visual input burden issue first; reduce that
  input load or inspect long-context handling before judging output quality.
- _Key signals:_ output/prompt=1.70%; visual input burden=93%; missing
  sections: title; missing terms: Bird, Coast, Deben, Woodbridge, GBR
- _Tokens:_ prompt 6592 tok; estimated text 478 tok; estimated non-text 6114
  tok; generated 112 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: GBR, bank; keywords=20; low-draft-improvement
- _Tokens:_ prompt 1749 tok; estimated text 478 tok; estimated non-text 1271
  tok; generated 114 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16779 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes
- _Tokens:_ prompt 16779 tok; estimated text 478 tok; estimated non-text 16301
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Deben, Woodbridge, GBR, sailing;
  keywords=19; low-draft-improvement
- _Tokens:_ prompt 3377 tok; estimated text 478 tok; estimated non-text 2899
  tok; generated 106 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: GBR, bank; keywords=20; low-draft-improvement
- _Tokens:_ prompt 1749 tok; estimated text 478 tok; estimated non-text 1271
  tok; generated 114 tok; requested max 500 tok; stop reason completed


### `Qwen/Qwen3-VL-2B-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16779 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes
- _Tokens:_ prompt 16779 tok; estimated text 478 tok; estimated non-text 16301
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes; excessive
  bullets=34
- _Tokens:_ prompt 1582 tok; estimated text 478 tok; estimated non-text 1104
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boating, Buoy, Bushes, Coast; low-draft-improvement
- _Tokens:_ prompt 544 tok; estimated text 478 tok; estimated non-text 66 tok;
  generated 134 tok; requested max 500 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a mixed burden issue first; reduce that input
  load or inspect long-context handling before judging output quality.
- _Key signals:_ Output is very short relative to prompt size (0.1%) with weak
  text signal 'truncated', suggesting possible early-stop or prompt-handling
  issues.; At mixed burden (16790 tokens), output stayed unusually short (14
  tokens; ratio 0.1%; weak text signal truncated).; output/prompt=0.08%; mixed
  burden=97%
- _Tokens:_ prompt 16790 tok; estimated text 478 tok; estimated non-text 16312
  tok; generated 14 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ornith-1.0-35B-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Mudflat
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 137 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boat, Boating, Bushes, Coast
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 113 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16781 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes
- _Tokens:_ prompt 16781 tok; estimated text 478 tok; estimated non-text 16303
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Estuary
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 99 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Forest;
  low-draft-improvement
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 93 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Landscape
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 105 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16790 tokens), output became repetitive.;
  hit token cap (500); missing sections: description, keywords; missing terms:
  Bird, Buoy, Bushes, Coast, Estuary
- _Tokens:_ prompt 16790 tok; estimated text 478 tok; estimated non-text 16312
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Foliage
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 131 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Mudflat
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 121 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  Mudflat, Peaceful, Riverbank, Woodbridge, GBR; keyword duplication=50%
- _Tokens:_ prompt 1564 tok; estimated text 478 tok; estimated non-text 1086
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Forest
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 109 tok; requested max 500 tok; stop reason completed

