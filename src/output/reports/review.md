<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

_Generated on 2026-05-21 23:12:48 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](../check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/nanoLLaVA-1.5-4bit`: ✅ B (75/100) | Desc 87 | Keywords 0 | 374.3 tps
- `qnguyen3/nanoLLaVA`: ✅ B (75/100) | Desc 90 | Keywords 0 | 114.8 tps
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ✅ B (75/100) | Desc 90 | Keywords 0 | 19.6 tps
- `mlx-community/MolmoPoint-8B-fp16`: ✅ B (73/100) | Desc 86 | Keywords 0 | 6.0 tps
- `mlx-community/FastVLM-0.5B-bf16`: ✅ B (68/100) | Desc 87 | Keywords 0 | 351.9 tps

### Watchlist

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | 494.7 tps | harness
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | 20.4 tps | harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 60 | Keywords 0 | 32.0 tps | degeneration, generation loop, harness
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (5/100) | Desc 44 | Keywords 0 | 40.1 tps | harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (5/100) | Desc 44 | Keywords 0 | 6.4 tps | harness

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                              | Verdict   | Hint Handling           | Key Evidence               |
|------------------------------------|-----------|-------------------------|----------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit` | `clean`   | preserves trusted hints | nontext prompt burden=73%  |
| `qnguyen3/nanoLLaVA`               | `clean`   | preserves trusted hints | nontext prompt burden=73%  |
| `mlx-community/MolmoPoint-8B-fp16` | `clean`   | preserves trusted hints | nontext prompt burden=100% |

### `caveat`

| Model                                               | Verdict          | Hint Handling           | Key Evidence                                                                                                                                                                                                                                                        |
|-----------------------------------------------------|------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/paligemma2-3b-pt-896-4bit`           | `context_budget` | preserves trusted hints | Output is very short relative to prompt size (0.2%), suggesting possible early-stop or prompt-handling issues. \| At long prompt length (4103 tokens), output stayed unusually short (9 tokens; ratio 0.2%). \| output/prompt=0.22% \| nontext prompt burden=100%   |
| `microsoft/Phi-3.5-vision-instruct`                 | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99%                                                                                                                                                                                                                    |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99%                                                                                                                                                                                                                    |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| degeneration=character_loop: ' ' repeated                                                                                                                                                                      |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=62% \| degeneration=repeated_punctuation: '##########...'                                                                                                                                                              |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=65% \| degeneration=repeated_punctuation: '##########...'                                                                                                                                                              |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| reasoning leak                                                                                                                                                                                                  |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `context_budget` | preserves trusted hints | Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues. \| At long prompt length (16167 tokens), output stayed unusually short (12 tokens; ratio 0.1%). \| output/prompt=0.07% \| nontext prompt burden=100% |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling           | Key Evidence                                                                                                                                                                                    |
|---------------------------------------------------------|---------------------|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `facebook/pe-av-large`                                  | `runtime_failure`   | not evaluated           | model error \| mlx lm model load model                                                                                                                                                          |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `runtime_failure`   | not evaluated           | model error \| mlx model load model                                                                                                                                                             |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `runtime_failure`   | not evaluated           | weight mismatch \| mlx model load weight mismatch                                                                                                                                               |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `harness`           | preserves trusted hints | Output appears truncated to about 3 tokens. \| nontext prompt burden=98%                                                                                                                        |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98% \| repetitive token=and                                                                                                                        |
| `mlx-community/FastVLM-0.5B-bf16`                       | `semantic_mismatch` | preserves trusted hints | nontext prompt burden=77%                                                                                                                                                                       |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `harness`           | preserves trusted hints | Output appears truncated to about 6 tokens. \| nontext prompt burden=99%                                                                                                                        |
| `mlx-community/InternVL3-8B-bf16`                       | `harness`           | preserves trusted hints | Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=100%                                                    |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98% \| repetitive token=phrase: "rew rew rew rew..."                                                                                               |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=94% \| repetitive token=phrase: "saff saff saff saff..."                                                                                           |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| repetitive token=phrase: ". . . ...."                                                                                                      |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| repetitive token=phrase: "ter budget ter budget..."                                                                                         |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `harness`           | preserves trusted hints | Output appears truncated to about 6 tokens. \| nontext prompt burden=99%                                                                                                                        |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| repetitive token=phrase: "ter budget ter budget..."                                                                                         |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98% \| degeneration=character_loop: '/1' repeated                                                                                                  |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| reasoning leak                                                                                                                              |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `semantic_mismatch` | preserves trusted hints | nontext prompt burden=99%                                                                                                                                                                       |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98%                                                                                                                                                |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| repetitive token=phrase: "1st 1st 1st 1st..."                                                                                              |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100%                                                                                                                                               |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| repetitive token=phrase: "out of course. out..."                                                                                           |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98% \| formatting=Unknown tags: &lt;td&gt;                                                                                                         |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| formatting=Unknown tags: &lt;fake_token_around_image&gt; \| repetitive token=comm,                                                         |
| `mlx-community/gemma-4-31b-it-4bit`                     | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98% \| degeneration=incomplete_sentence: ends with 'la'                                                                                            |
| `mlx-community/InternVL3-14B-8bit`                      | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| repetitive token=phrase: "2018 2018 2018 2018..."                                                                                          |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | preserves trusted hints | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 41 occurrences). \| hit token cap (200) \| nontext prompt burden=100% \| degeneration=character_loop: '.99' repeated |
| `mlx-community/pixtral-12b-8bit`                        | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| repetitive token=phrase: "the other the other..."                                                                                          |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| repetitive token=phrase: "and the idea is..."                                                                                              |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| repetitive token=phrase: "1.0, 1.0, 1.0, 1.0,..."                                                                                          |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100%                                                                                                                                               |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98%                                                                                                                                                |
| `mlx-community/pixtral-12b-bf16`                        | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100%                                                                                                                                               |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `harness`           | preserves trusted hints | Output appears truncated to about 3 tokens. \| nontext prompt burden=100%                                                                                                                       |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| repetitive token=phrase: "the image shows a..."                                                                                            |
| `mlx-community/X-Reasoner-7B-8bit`                      | `harness`           | preserves trusted hints | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (200) \| nontext prompt burden=100%                                                                    |
| `mlx-community/GLM-4.6V-nvfp4`                          | `semantic_mismatch` | preserves trusted hints | nontext prompt burden=100% \| degeneration=incomplete_sentence: ends with 'co'                                                                                                                  |
| `mlx-community/gemma-4-31b-bf16`                        | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98%                                                                                                                                                |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100%                                                                                                                                               |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | preserves trusted hints | At long prompt length (16167 tokens), output became repetitive. \| hit token cap (200) \| nontext prompt burden=100% \| repetitive token=all,                                                   |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `harness`           | preserves trusted hints | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (200) \| nontext prompt burden=100%                                                                    |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | preserves trusted hints | At long prompt length (16167 tokens), output became repetitive. \| hit token cap (200) \| nontext prompt burden=100% \| degeneration=character_loop: ' ,' repeated                              |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| degeneration=character_loop: '00' repeated                                                                                                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| degeneration=character_loop: '66' repeated                                                                                                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints | At long prompt length (16167 tokens), output became repetitive. \| hit token cap (200) \| nontext prompt burden=100% \| degeneration=incomplete_sentence: ends with '2v'                        |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                     | Affected Models                                            | Issue Draft                                                                    | Evidence Bundle   | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                                         | 1: `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)             | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx`                                                    | Weight/config mismatch during model load              | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                     | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx_mlx-model-load-weight-mismatch_001.md)   | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-lm`                                                 | Missing module/import during model load               | Model Error \| phase model_load \| ModuleNotFoundError                                                                                                                                | 1: `facebook/pe-av-large`                                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-lm_mlx-lm-model-load-model_001.md)       | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers              | 41 BPE space markers found in decoded text \| prompt=1,745 \| output/prompt=11.46% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200)                                   | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_encoding_001.md)                     | -                 | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=16,176 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) \| 2 model cluster | 2: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+1)          | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm_stop-token_001.md)                   | -                 | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~3 \| prompt=269 \| output/prompt=1.12% \| nontext burden=98% \| stop=completed \| 5 model cluster                                                                    | 5: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (+4)                 | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_model-config-mlx-vlm_prompt-template_001.md) | -                 | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | prompt_tokens=16167, repetitive output \| prompt=16,167 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) \| 3 model cluster                    | 3: `mlx-community/Qwen3.5-27B-mxfp8` (+2)                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_007_mlx-vlm-mlx_long-context_001.md)             | -                 | Full and reduced reruns avoid context collapse.           |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | output/prompt=0.1% \| prompt_tokens=16167, output_tokens=12, output/prompt=0.1% \| prompt=16,167 \| output/prompt=0.07% \| nontext burden=100% \| stop=completed \| 2 model cluster   | 2: `mlx-community/Qwen3.5-9B-MLX-4bit` (+1)                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_008_mlx-vlm-mlx_long-context_002.md)             | -                 | Full and reduced reruns avoid context collapse.           |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | token cap \| abrupt tail \| degeneration \| prompt=16,167 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) \| 2 model cluster                  | 2: `mlx-community/Qwen3.5-27B-4bit` (+1)                   | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_009_mlx-vlm-mlx_long-context_003.md)             | -                 | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

## Model Verdicts

### `facebook/pe-av-large`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx-lm`; reported package `mlx-lm`; failure stage
  `Model Error`; diagnostic code `MLX_LM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect the import path and installed package version that owns
  the missing module before treating this as a model failure.
- _Key signals:_ model error; mlx lm model load model
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Model
  Error`; diagnostic code `MLX_MODEL_LOAD_MODEL`
- _Next step:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.
- _Key signals:_ model error; mlx model load model
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Weight
  Mismatch`; diagnostic code `MLX_MODEL_LOAD_WEIGHT_MISMATCH`
- _Next step:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.
- _Key signals:_ weight mismatch; mlx model load weight mismatch
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.; nontext prompt
  burden=98%
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 3 tok; requested max 200 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 99 tok; requested max 200 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; repetitive
  token=and
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=77%
- _Tokens:_ prompt 26 tok; estimated text 6 tok; estimated non-text 20 tok;
  generated 35 tok; requested max 200 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 71 tok; requested max 200 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 6 tokens.; nontext prompt
  burden=99%
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 6 tok; requested max 200 tok; stop reason completed


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.
- _Key signals:_ Output is very short relative to prompt size (0.2%),
  suggesting possible early-stop or prompt-handling issues.; At long prompt
  length (4103 tokens), output stayed unusually short (9 tokens; ratio 0.2%).;
  output/prompt=0.22%; nontext prompt burden=100%
- _Tokens:_ prompt 4103 tok; estimated text 6 tok; estimated non-text 4097
  tok; generated 9 tok; requested max 200 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (0.5%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=100%
- _Tokens:_ prompt 2317 tok; estimated text 6 tok; estimated non-text 2311
  tok; generated 11 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; repetitive
  token=phrase: "rew rew rew rew..."
- _Tokens:_ prompt 266 tok; estimated text 6 tok; estimated non-text 260 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=94%; repetitive
  token=phrase: "saff saff saff saff..."
- _Tokens:_ prompt 97 tok; estimated text 6 tok; estimated non-text 91 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: ". . . ...."
- _Tokens:_ prompt 2277 tok; estimated text 6 tok; estimated non-text 2271
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; repetitive
  token=phrase: "ter budget ter budget..."
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 6 tokens.; nontext prompt
  burden=99%
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 6 tok; requested max 200 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; repetitive
  token=phrase: "ter budget ter budget..."
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  degeneration=character_loop: '/1' repeated
- _Tokens:_ prompt 284 tok; estimated text 6 tok; estimated non-text 278 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; reasoning
  leak
- _Tokens:_ prompt 1033 tok; estimated text 6 tok; estimated non-text 1027
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 76 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "1st 1st 1st 1st..."
- _Tokens:_ prompt 2278 tok; estimated text 6 tok; estimated non-text 2272
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Tokens:_ prompt 1270 tok; estimated text 6 tok; estimated non-text 1264
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "out of course. out..."
- _Tokens:_ prompt 1866 tok; estimated text 6 tok; estimated non-text 1860
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  formatting=Unknown tags: &lt;td&gt;
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  formatting=Unknown tags: &lt;fake_token_around_image&gt;; repetitive
  token=comm,
- _Tokens:_ prompt 2327 tok; estimated text 6 tok; estimated non-text 2321
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  degeneration=incomplete_sentence: ends with 'la'
- _Tokens:_ prompt 284 tok; estimated text 6 tok; estimated non-text 278 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "2018 2018 2018 2018..."
- _Tokens:_ prompt 2317 tok; estimated text 6 tok; estimated non-text 2311
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=character_loop: '  ' repeated
- _Tokens:_ prompt 2278 tok; estimated text 6 tok; estimated non-text 2272
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 41 occurrences).; hit token cap (200); nontext prompt
  burden=100%; degeneration=character_loop: '.99' repeated
- _Tokens:_ prompt 1745 tok; estimated text 6 tok; estimated non-text 1739
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "the other the other..."
- _Tokens:_ prompt 2349 tok; estimated text 6 tok; estimated non-text 2343
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "and the idea is..."
- _Tokens:_ prompt 6045 tok; estimated text 6 tok; estimated non-text 6039
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=62%;
  degeneration=repeated_punctuation: '##########...'
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "1.0, 1.0, 1.0, 1.0,..."
- _Tokens:_ prompt 6045 tok; estimated text 6 tok; estimated non-text 6039
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Tokens:_ prompt 2440 tok; estimated text 6 tok; estimated non-text 2434
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Tokens:_ prompt 2349 tok; estimated text 6 tok; estimated non-text 2343
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.; nontext prompt
  burden=100%
- _Tokens:_ prompt 1201 tok; estimated text 6 tok; estimated non-text 1195
  tok; generated 3 tok; requested max 200 tok; stop reason completed


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 2818 tok; estimated text 6 tok; estimated non-text 2812
  tok; generated 107 tok; requested max 200 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "the image shows a..."
- _Tokens:_ prompt 1201 tok; estimated text 6 tok; estimated non-text 1195
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; hit token cap (200); nontext prompt burden=100%
- _Tokens:_ prompt 16176 tok; estimated text 6 tok; estimated non-text 16170
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%; degeneration=incomplete_sentence:
  ends with 'co'
- _Tokens:_ prompt 6045 tok; estimated text 6 tok; estimated non-text 6039
  tok; generated 58 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Tokens:_ prompt 272 tok; estimated text 6 tok; estimated non-text 266 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=65%;
  degeneration=repeated_punctuation: '##########...'
- _Tokens:_ prompt 17 tok; estimated text 6 tok; estimated non-text 11 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; reasoning
  leak
- _Tokens:_ prompt 1033 tok; estimated text 6 tok; estimated non-text 1027
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.
- _Key signals:_ Output is very short relative to prompt size (0.1%),
  suggesting possible early-stop or prompt-handling issues.; At long prompt
  length (16167 tokens), output stayed unusually short (12 tokens; ratio
  0.1%).; output/prompt=0.07%; nontext prompt burden=100%
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 12 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16167 tokens), output became
  repetitive.; hit token cap (200); nontext prompt burden=100%; repetitive
  token=all,
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; hit token cap (200); nontext prompt burden=100%
- _Tokens:_ prompt 16176 tok; estimated text 6 tok; estimated non-text 16170
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16167 tokens), output became
  repetitive.; hit token cap (200); nontext prompt burden=100%;
  degeneration=character_loop: ' ,' repeated
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=character_loop: '00' repeated
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=character_loop: '66' repeated
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16167 tokens), output became
  repetitive.; hit token cap (200); nontext prompt burden=100%;
  degeneration=incomplete_sentence: ends with '2v'
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

