<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

Generated on: 2026-07-21 08:38:01 BST

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Companion artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Qwen3.5-27B-mxfp8`: 🏆 A (89/100) | Desc 100 | Keywords 76 | Δ+16 | 17.5 tps
- `mlx-community/Qwen3.5-27B-4bit`: 🏆 A (88/100) | Desc 100 | Keywords 77 | Δ+16 | 31.0 tps
- `mlx-community/Qwen3.5-35B-A3B-6bit`: 🏆 A (87/100) | Desc 100 | Keywords 84 | Δ+14 | 92.5 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (86/100) | Desc 100 | Keywords 91 | Δ+14 | 66.9 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (86/100) | Desc 100 | Keywords 77 | Δ+13 | 63.6 tps

### Watchlist

- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-73 | 45281.6 tps | harness
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (6/100) | Desc 45 | Keywords 0 | Δ-67 | 62.5 tps | context ignored, harness, long context, low metadata alignment
- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-53 | 93.3 tps | cutoff, degeneration, generation loop, harness, long context, low metadata alignment, missing sections, repetitive, text sanity
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (20/100) | Desc 40 | Keywords 0 | Δ-53 | 91.4 tps | cutoff, degeneration, generation loop, harness, long context, low metadata alignment, missing sections, repetitive, text sanity
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟡 C (50/100) | Desc 42 | Keywords 45 | Δ-22 | 221.7 tps | context ignored, cutoff, generation loop, harness, long context, low metadata alignment, missing sections, repetitive, unverified context copy

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                                   | Verdict          | Hint Handling           | Key Evidence                   |
|---------------------------------------------------------|------------------|-------------------------|--------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | `clean`          | preserves trusted hints | Empty thinking wrapper present |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/FastVLM-0.5B-bf16`                       | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `clean`          | preserves trusted hints | no flagged signals             |
| `microsoft/Phi-3.5-vision-instruct`                     | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/InternVL3-8B-bf16`                       | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/pixtral-12b-8bit`                        | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/gemma-4-31b-it-4bit`                     | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/InternVL3-14B-8bit`                      | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/pixtral-12b-bf16`                        | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/MolmoPoint-8B-fp16`                      | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Ornith-1.0-35B-bf16`                     | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `context_budget` | preserves trusted hints | no flagged signals             |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `clean`          | preserves trusted hints | no flagged signals             |
| `mlx-community/Qwen3.5-27B-4bit`                        | `clean`          | preserves trusted hints | no flagged signals             |

### `caveat`

| Model                                      | Verdict             | Hint Handling           | Key Evidence                                                                                                                                                                                                  |
|--------------------------------------------|---------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16`        | `model_shortcoming` | preserves trusted hints | irrelevant output smell                                                                                                                                                                                       |
| `mlx-community/Idefics3-8B-Llama3-bf16`    | `clean`             | preserves trusted hints | special token wrapper \| Unknown tags: <end_of_utterance>                                                                                                                                                     |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `model_shortcoming` | preserves trusted hints | irrelevant output smell                                                                                                                                                                                       |
| `mlx-community/GLM-4.6V-Flash-6bit`        | `clean`             | preserves trusted hints | special token wrapper                                                                                                                                                                                         |
| `mlx-community/GLM-4.6V-nvfp4`             | `clean`             | preserves trusted hints | special token wrapper                                                                                                                                                                                         |
| `mlx-community/X-Reasoner-7B-8bit`         | `context_budget`    | no overlap              | Output appears truncated to about 9 tokens. \| At mixed burden (16933 tokens), output stayed unusually short (9 tokens; ratio 0.1%; weak text signal truncated). \| special token wrapper \| harness contract |
| `mlx-community/Molmo-7B-D-0924-bf16`       | `harness`           | not evaluated           | harness contract                                                                                                                                                                                              |

### `needs_triage`

- None.

### `avoid`

| Model                                              | Verdict             | Hint Handling           | Key Evidence                                                                                                                  |
|----------------------------------------------------|---------------------|-------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`                 | `model_shortcoming` | preserves trusted hints | missing required sections \| irrelevant output smell \| missing sections: keywords                                            |
| `HuggingFaceTB/SmolVLM-Instruct`                   | `model_shortcoming` | preserves trusted hints | missing required sections \| irrelevant output smell \| missing sections: keywords                                            |
| `mlx-community/SmolVLM-Instruct-bf16`              | `model_shortcoming` | preserves trusted hints | missing required sections \| irrelevant output smell \| missing sections: keywords                                            |
| `qnguyen3/nanoLLaVA`                               | `model_shortcoming` | preserves trusted hints | missing required sections \| irrelevant output smell \| missing sections: keywords                                            |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`   | `model_shortcoming` | no overlap              | missing required sections \| irrelevant output smell \| missing sections: title, description, keywords                        |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`    | `model_shortcoming` | preserves trusted hints | missing required sections \| irrelevant output smell \| missing sections: title \| Unknown tags: <channel\|>                  |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`     | `model_shortcoming` | preserves trusted hints | missing required sections \| irrelevant output smell \| missing sections: title \| Unknown tags: <channel\|>                  |
| `mlx-community/gemma-3n-E2B-4bit`                  | `cutoff_degraded`   | not evaluated           | text repetition \| missing required sections \| token cap truncation \| hit token cap (500)                                   |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`   | `model_shortcoming` | no overlap              | missing required sections \| irrelevant output smell \| missing sections: title, description, keywords                        |
| `mlx-community/gemma-3n-E4B-it-bf16`               | `model_shortcoming` | preserves trusted hints | missing required sections \| irrelevant output smell \| missing sections: title, description, keywords                        |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`            | `cutoff_degraded`   | preserves trusted hints | thinking trace \| reasoning budget exhausted \| missing required sections \| hit token cap (500)                              |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`          | `cutoff_degraded`   | preserves trusted hints | thinking trace \| reasoning budget exhausted \| missing required sections \| hit token cap (500)                              |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff_degraded`   | preserves trusted hints | missing required sections \| token cap truncation \| hit token cap (500) \| missing sections: title, description, keywords    |
| `mlx-community/paligemma2-3b-pt-896-4bit`          | `cutoff_degraded`   | no overlap              | text repetition \| missing required sections \| token cap truncation \| hit token cap (500)                                   |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`    | `cutoff_degraded`   | preserves trusted hints | missing required sections \| token cap truncation \| hit token cap (500) \| missing sections: description, keywords           |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`          | `context_budget`    | preserves trusted hints | missing required sections \| missing sections: title, description, keywords                                                   |
| `Qwen/Qwen3-VL-2B-Instruct`                        | `cutoff_degraded`   | not evaluated           | At mixed burden (16922 tokens), output became repetitive. \| harness contract \| text repetition \| missing required sections |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`         | `model_shortcoming` | preserves trusted hints | missing required sections \| irrelevant output smell \| missing sections: title, description, keywords                        |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`          | `cutoff_degraded`   | not evaluated           | At mixed burden (16922 tokens), output became repetitive. \| harness contract \| text repetition \| missing required sections |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`    | `cutoff_degraded`   | no overlap              | text repetition \| missing required sections \| token cap truncation \| hit token cap (500)                                   |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`          | `cutoff_degraded`   | no overlap              | At mixed burden (16933 tokens), output became repetitive. \| harness contract \| text repetition \| missing required sections |
| `mlx-community/gemma-4-31b-bf16`                   | `cutoff_degraded`   | preserves trusted hints | text repetition \| missing required sections \| token cap truncation \| hit token cap (500)                                   |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`     | `cutoff_degraded`   | preserves trusted hints | thinking trace \| reasoning budget exhausted \| text repetition \| hit token cap (500)                                        |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                                                                                                                                                               | Evidence Snapshot                                                                                                                                                                 | Affected Models                         | Issue Draft                                                                                                                                                              | Evidence Bundle                                                                                                                                                                                | Fixed When                                                |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| model configuration / repository                         | Model config: Processor load / processor error: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor. | Processor Error \| phase processor_load \| ValueError                                                                                                                             | 1: `mlx-community/Step-3.7-Flash-oQ2e`  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_model-configuration-repository_model-config-processor-load-processor_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260721T073757Z_001_mlx-community_Step-3.7-Flash-oQ2e_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_3c.json)  | Load/generation completes or fails with a narrower owner. |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                                                                                                                                                 | harness:prompt template \| prompt=1,788 \| output/prompt=0.06% \| stop=completed                                                                                                  | 1: `mlx-community/Molmo-7B-D-0924-bf16` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_model-config-mlx-vlm_prompt-template_001.md)                                 | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260721T073757Z_004_mlx-community_Molmo-7B-D-0924-bf16_model_config_mlx_vlm_prompt_template_001.json) | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short                                                                                                                                 | prompt_tokens=16922, repetitive output \| prompt=16,922 \| output/prompt=2.95% \| mixed burden=97% \| stop=max_tokens \| hit token cap (500) \| 3 model cluster                   | 3: `Qwen/Qwen3-VL-2B-Instruct` (+2)     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm-mlx_long-context_001.md)                                             | [3 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260721T073757Z_003_Qwen_Qwen3-VL-2B-Instruct_mlx_vlm_mlx_long_context_001.json)                   | Full and reduced reruns avoid context collapse.           |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short                                                                                                                                 | generated_tokens~9 \| prompt_tokens=16933, output_tokens=9, output/prompt=0.1%, weak text=truncated \| prompt=16,933 \| output/prompt=0.05% \| mixed burden=97% \| stop=completed | 1: `mlx-community/X-Reasoner-7B-8bit`   | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm-mlx_long-context_002.md)                                             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260721T073757Z_002_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_002.json)               | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

## Model Verdicts

### `mlx-community/Step-3.7-Flash-oQ2e`

- _Verdict:_ runtime_failure | user=not_evaluated
- _Why:_ execution failure
- _Maintainer:_ issue_ready | owner=model-config | confidence=medium
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.


### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/MiniCPM-V-4.6-8bit`

- _Verdict:_ clean | user=recommended
- _Why:_ Empty thinking wrapper present
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `HuggingFaceTB/SmolVLM-Instruct`

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/SmolVLM-Instruct-bf16`

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `qnguyen3/nanoLLaVA`

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Verdict:_ model_shortcoming | user=caveat
- _Why:_ irrelevant output smell
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/FastVLM-0.5B-bf16`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title, description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title | Unknown tags: <channel|>
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/diffusiongemma-26B-A4B-it-8bit`

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title | Unknown tags: <channel|>
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `microsoft/Phi-3.5-vision-instruct`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/InternVL3-8B-bf16`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/gemma-3n-E2B-4bit`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ text repetition | missing required sections | token cap truncation |
  hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title, description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/pixtral-12b-8bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/gemma-4-31b-it-4bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Verdict:_ clean | user=caveat
- _Why:_ special token wrapper | Unknown tags: <end_of_utterance>
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/InternVL3-14B-8bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Verdict:_ model_shortcoming | user=caveat
- _Why:_ irrelevant output smell
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title, description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Verdict:_ clean | user=caveat
- _Why:_ special token wrapper
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/pixtral-12b-bf16`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ thinking trace | reasoning budget exhausted | missing required
  sections | hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ thinking trace | reasoning budget exhausted | missing required
  sections | hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ missing required sections | token cap truncation | hit token cap
  (500) | missing sections: title, description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ text repetition | missing required sections | token cap truncation |
  hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ missing required sections | token cap truncation | hit token cap
  (500) | missing sections: description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/GLM-4.6V-nvfp4`

- _Verdict:_ clean | user=caveat
- _Why:_ special token wrapper
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/X-Reasoner-7B-8bit`

- _Verdict:_ context_budget | user=caveat
- _Why:_ Output appears truncated to about 9 tokens. | At mixed burden (16933
  tokens), output stayed unusually short (9 tokens; ratio 0.1%; weak text
  signal truncated). | special token wrapper | harness contract
- _Maintainer:_ issue_ready | owner=mlx | confidence=high
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.


### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- _Verdict:_ context_budget | user=avoid
- _Why:_ missing required sections | missing sections: title, description,
  keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `Qwen/Qwen3-VL-2B-Instruct`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ At mixed burden (16922 tokens), output became repetitive. | harness
  contract | text repetition | missing required sections
- _Maintainer:_ issue_ready | owner=mlx | confidence=high
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title, description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Verdict:_ harness | user=caveat
- _Why:_ harness contract
- _Maintainer:_ issue_ready | owner=model-config | confidence=high
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ At mixed burden (16922 tokens), output became repetitive. | harness
  contract | text repetition | missing required sections
- _Maintainer:_ issue_ready | owner=mlx | confidence=high
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.


### `mlx-community/MolmoPoint-8B-fp16`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ text repetition | missing required sections | token cap truncation |
  hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Ornith-1.0-35B-bf16`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Verdict:_ context_budget | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ At mixed burden (16933 tokens), output became repetitive. | harness
  contract | text repetition | missing required sections
- _Maintainer:_ issue_ready | owner=mlx | confidence=high
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.


### `mlx-community/gemma-4-31b-bf16`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ text repetition | missing required sections | token cap truncation |
  hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Qwen3.5-27B-4bit`

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ thinking trace | reasoning budget exhausted | text repetition | hit
  token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

