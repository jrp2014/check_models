<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

Generated on: 2026-07-05 22:43:04 BST

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Companion artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: caption-review candidate | 129.0 tps
- `mlx-community/GLM-4.6V-Flash-6bit`: caption-review candidate | 63.9 tps
- `mlx-community/GLM-4.6V-nvfp4`: caption-review candidate | 51.2 tps
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: caption-review candidate | 34.7 tps
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: caption-review candidate | 19.6 tps

### Watchlist

- `mlx-community/X-Reasoner-7B-8bit`: caption-review candidate | 88.4 tps | harness
- `mlx-community/paligemma2-3b-pt-896-4bit`: caption-review candidate | 78.7 tps | harness, long context
- `mlx-community/Qwen3.6-27B-mxfp8`: caption-review candidate | 36.1 tps | harness
- `mlx-community/Qwen3.5-35B-A3B-bf16`: caption-review candidate | 77.1 tps | harness
- `Qwen/Qwen3-VL-2B-Instruct`: caption-review candidate | 148.9 tps | harness

## User Buckets

User-first summary grouped by recommendation bucket.

### `clean-triage-pass`

Clean in the current ungrounded triage run; visual correctness still needs gallery review or grounded metadata.

| Model                                                   | Verdict     | Hint Handling   | Key Evidence                                      |
|---------------------------------------------------------|-------------|-----------------|---------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/FastVLM-0.5B-bf16`                       | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `clean`     | not evaluated   | no flagged signals                                |
| `qnguyen3/nanoLLaVA`                                    | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `clean`     | not evaluated   | no flagged signals                                |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `clean`     | not evaluated   | no flagged signals                                |
| `microsoft/Phi-3.5-vision-instruct`                     | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen3.5-27B-4bit`                        | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-4-31b-it-4bit`                     | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `clean`     | not evaluated   | formatting=Unknown tags: &lt;end_of_utterance&gt; |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/InternVL3-8B-bf16`                       | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/InternVL3-14B-8bit`                      | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `clean`     | not evaluated   | no flagged signals                                |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `token_cap` | not evaluated   | hit token cap (200) \| reasoning leak             |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Ornith-1.0-35B-bf16`                     | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/GLM-4.6V-nvfp4`                          | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-4-31b-bf16`                        | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `clean`     | not evaluated   | no flagged signals                                |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/MolmoPoint-8B-fp16`                      | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `clean`     | not evaluated   | no flagged signals                                |

### `caveat`

| Model                                     | Verdict          | Hint Handling   | Key Evidence                                                                                                                                                                                                               |
|-------------------------------------------|------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/paligemma2-3b-pt-896-4bit` | `context_budget` | not evaluated   | Output appears truncated to about 3 tokens. \| At long prompt length (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%; weak text signal truncated). \| output/prompt=0.07% \| nontext prompt burden=100% |

### `needs_triage`

- None.

### `avoid`

| Model                                              | Verdict             | Hint Handling   | Key Evidence                                                                                                              |
|----------------------------------------------------|---------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------|
| `Qwen/Qwen3-VL-2B-Instruct`                        | `harness`           | not evaluated   | Output appears truncated to about 3 tokens.                                                                               |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`          | `harness`           | not evaluated   | Output appears truncated to about 3 tokens.                                                                               |
| `mlx-community/X-Reasoner-7B-8bit`                 | `harness`           | not evaluated   | Output appears truncated to about 3 tokens.                                                                               |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`          | `harness`           | not evaluated   | Special control token &lt;/think&gt; appeared in generated text. \| text-sanity=gibberish(token_noise)                    |
| `mlx-community/Qwen3.6-27B-mxfp8`                  | `harness`           | not evaluated   | Output appears truncated to about 2 tokens.                                                                               |
| `mlx-community/gemma-3n-E2B-4bit`                  | `cutoff_degraded`   | not evaluated   | hit token cap (200) \| repetitive token=phrase: "have this image. have..."                                                |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                | `semantic_mismatch` | not evaluated   | text-sanity=gibberish(mixed_script_noise)                                                                                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`          | `cutoff_degraded`   | not evaluated   | hit token cap (200) \| reasoning leak                                                                                     |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`    | `harness`           | not evaluated   | Special control token &lt;\|end\|&gt; appeared in generated text. \| reasoning leak \| text-sanity=gibberish(token_noise) |
| `mlx-community/Qwen3.5-35B-A3B-bf16`               | `harness`           | not evaluated   | Output appears truncated to about 3 tokens.                                                                               |
| `mlx-community/pixtral-12b-8bit`                   | `cutoff_degraded`   | not evaluated   | hit token cap (200)                                                                                                       |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff_degraded`   | not evaluated   | hit token cap (200) \| degeneration=incomplete_sentence: ends with 'a'                                                    |
| `mlx-community/pixtral-12b-bf16`                   | `cutoff_degraded`   | not evaluated   | hit token cap (200)                                                                                                       |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`     | `cutoff_degraded`   | not evaluated   | hit token cap (200) \| reasoning leak                                                                                     |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                  | Affected Models                                         | Issue Draft                                                                                                                              | Evidence Bundle                                                                                                                                                                            | Fixed When                                          |
|----------------------------------------------------------|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;\|end\|&gt; \| prompt=1,330 \| output/prompt=13.08% \| nontext burden=100% \| stop=completed \| 2 model cluster                            | 2: `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (+1) | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx-vlm_stop-token_001.md)                   | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T214304Z_002_mlx-community_Apriel-1.5-15b-Thinker-6bit-MLX_mlx_vlm_stop_token_001.json) | No leaked stop/control tokens.                      |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~3 \| prompt=315 \| output/prompt=0.95% \| nontext burden=98% \| stop=completed \| 5 model cluster                                                                 | 5: `Qwen/Qwen3-VL-2B-Instruct` (+4)                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_model-config-mlx-vlm_prompt-template_001.md) | [5 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T214304Z_001_Qwen_Qwen3-VL-2B-Instruct_model_config_mlx_vlm_prompt_template_001.json)   | Requested sections render without template leakage. |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | generated_tokens~3 \| prompt_tokens=4103, output_tokens=3, output/prompt=0.1%, weak text=truncated \| prompt=4,103 \| output/prompt=0.07% \| nontext burden=100% \| stop=completed | 1: `mlx-community/paligemma2-3b-pt-896-4bit`            | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm-mlx_long-context_001.md)             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T214304Z_008_mlx-community_paligemma2-3b-pt-896-4bit_mlx_vlm_mlx_long_context_001.json)    | Full and reduced reruns avoid context collapse.     |
<!-- markdownlint-enable MD060 -->

## Model Verdicts

### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 80 tok; estimated text 6 tok; estimated non-text 74 tok;
  generated 10 tok; requested max 200 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 10 tok; requested max 200 tok; stop reason completed


### `Qwen/Qwen3-VL-2B-Instruct`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 3 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 3 tok; requested max 200 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 26 tok; estimated text 6 tok; estimated non-text 20 tok;
  generated 15 tok; requested max 200 tok; stop reason completed


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 13 tok; requested max 200 tok; stop reason completed


### `mlx-community/MiniCPM-V-4.6-8bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 228 tok; estimated text 6 tok; estimated non-text 222 tok;
  generated 22 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 25 tok; requested max 200 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 81 tok; requested max 200 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 3 tok; requested max 200 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 35 tok; requested max 200 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 19 tok; requested max 200 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 19 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 24 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; text-sanity=gibberish(token_noise)
- _Tokens:_ prompt 317 tok; estimated text 6 tok; estimated non-text 311 tok;
  generated 76 tok; requested max 200 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 973 tok; estimated text 6 tok; estimated non-text 967 tok;
  generated 98 tok; requested max 200 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 42 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 30 tok; requested max 200 tok; stop reason completed


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 3 tokens.; At long prompt
  length (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%;
  weak text signal truncated).; output/prompt=0.07%; nontext prompt
  burden=100%
- _Tokens:_ prompt 4103 tok; estimated text 6 tok; estimated non-text 4097
  tok; generated 3 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 20 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 50 tok; requested max 200 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-8bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 21 tok; requested max 200 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 16 tok; requested max 200 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 974 tok; estimated text 6 tok; estimated non-text 968 tok;
  generated 50 tok; requested max 200 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 69 tok; requested max 200 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 974 tok; estimated text 6 tok; estimated non-text 968 tok;
  generated 60 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 2 tokens.
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 2 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 25 tok; requested max 200 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ formatting=Unknown tags: &lt;end_of_utterance&gt;
- _Tokens:_ prompt 2327 tok; estimated text 6 tok; estimated non-text 2321
  tok; generated 23 tok; requested max 200 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 97 tok; estimated text 6 tok; estimated non-text 91 tok;
  generated 187 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); repetitive token=phrase: "have this
  image. have..."
- _Tokens:_ prompt 266 tok; estimated text 6 tok; estimated non-text 260 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ text-sanity=gibberish(mixed_script_noise)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 166 tok; requested max 200 tok; stop reason completed


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 25 tok; requested max 200 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 3341 tok; estimated text 6 tok; estimated non-text 3335
  tok; generated 48 tok; requested max 200 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 2356 tok; estimated text 6 tok; estimated non-text 2350
  tok; generated 55 tok; requested max 200 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 3341 tok; estimated text 6 tok; estimated non-text 3335
  tok; generated 15 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 124 tok; requested max 200 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ clean triage pass; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); reasoning leak
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); reasoning leak
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 441 tok; estimated text 6 tok; estimated non-text 435 tok;
  generated 84 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 91 tok; requested max 200 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 101 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 70 tok; requested max 200 tok; stop reason completed


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; reasoning leak; text-sanity=gibberish(token_noise)
- _Tokens:_ prompt 1330 tok; estimated text 6 tok; estimated non-text 1324
  tok; generated 174 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 3 tok; requested max 200 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200)
- _Tokens:_ prompt 1239 tok; estimated text 6 tok; estimated non-text 1233
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 55 tok; requested max 200 tok; stop reason completed


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 119 tok; requested max 200 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 53 tok; requested max 200 tok; stop reason completed


### `mlx-community/Ornith-1.0-35B-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 44 tok; requested max 200 tok; stop reason completed


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 71 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 36 tok; requested max 200 tok; stop reason completed


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); degeneration=incomplete_sentence: ends
  with 'a'
- _Tokens:_ prompt 439 tok; estimated text 6 tok; estimated non-text 433 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 85 tok; requested max 200 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200)
- _Tokens:_ prompt 1239 tok; estimated text 6 tok; estimated non-text 1233
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 17 tok; estimated text 6 tok; estimated non-text 11 tok;
  generated 77 tok; requested max 200 tok; stop reason completed


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 766 tok; estimated text 6 tok; estimated non-text 760 tok;
  generated 163 tok; requested max 200 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ clean triage pass; review verdict: clean
- _Owner:_ likely owner `model`
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 159 tok; requested max 200 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); reasoning leak
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

