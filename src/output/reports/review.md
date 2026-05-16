# Automated Review Digest

_Generated on 2026-05-17 00:16:46 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/nanoLLaVA-1.5-4bit`: ✅ B (75/100) | Desc 87 | Keywords 0 | 324.0 tps
- `qnguyen3/nanoLLaVA`: ✅ B (75/100) | Desc 90 | Keywords 0 | 115.1 tps
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ✅ B (75/100) | Desc 90 | Keywords 0 | 19.6 tps

### Watchlist

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | 551.0 tps | harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | 43.6 tps | harness
- `mlx-community/GLM-4.6V-nvfp4`: ❌ F (5/100) | Desc 22 | Keywords 0 | 47.5 tps | harness
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (5/100) | Desc 44 | Keywords 0 | 39.6 tps | harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (5/100) | Desc 44 | Keywords 0 | 6.4 tps | harness

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                           | Verdict   | Hint Handling           | Key Evidence               |
|-------------------------------------------------|-----------|-------------------------|----------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`              | `clean`   | preserves trusted hints | nontext prompt burden=73%  |
| `qnguyen3/nanoLLaVA`                            | `clean`   | preserves trusted hints | nontext prompt burden=73%  |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `clean`   | preserves trusted hints | nontext prompt burden=99%  |
| `mlx-community/Molmo-7B-D-0924-8bit`            | `clean`   | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/Molmo-7B-D-0924-bf16`            | `clean`   | preserves trusted hints | nontext prompt burden=100% |

### `caveat`

| Model                                              | Verdict          | Hint Handling           | Key Evidence                                                                                                                                                                                                                                                      |
|----------------------------------------------------|------------------|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/paligemma2-3b-pt-896-4bit`          | `context_budget` | preserves trusted hints | Output is very short relative to prompt size (0.2%), suggesting possible early-stop or prompt-handling issues. \| At long prompt length (4103 tokens), output stayed unusually short (9 tokens; ratio 0.2%). \| output/prompt=0.22% \| nontext prompt burden=100% |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`  | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100%                                                                                                                                                                                                                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`            | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98%                                                                                                                                                                                                                  |
| `mlx-community/Qwen3.5-35B-A3B-4bit`               | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| degeneration=character_loop: 'T;' repeated                                                                                                                                                                    |
| `mlx-community/GLM-4.6V-Flash-mxfp4`               | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| degeneration=character_loop: '13-' repeated                                                                                                                                                                   |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`    | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100%                                                                                                                                                                                                                 |
| `mlx-community/gemma-4-31b-it-4bit`                | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98% \| degeneration=character_loop: 'e_' repeated                                                                                                                                                                    |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `token_cap`      | preserves trusted hints | hit token cap (200) \| nontext prompt burden=62% \| degeneration=repeated_punctuation: '##########...'                                                                                                                                                            |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict           | Hint Handling           | Key Evidence                                                                                                                                                                    |
|---------------------------------------------------------|-------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `facebook/pe-av-large`                                  | `runtime_failure` | not evaluated           | model error \| mlx lm model load model                                                                                                                                          |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `runtime_failure` | not evaluated           | model error \| mlx model load model                                                                                                                                             |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `runtime_failure` | not evaluated           | weight mismatch \| mlx model load weight mismatch                                                                                                                               |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `harness`         | preserves trusted hints | Output appears truncated to about 4 tokens. \| nontext prompt burden=98%                                                                                                        |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98% \| repetitive token=and                                                                                                        |
| `mlx-community/InternVL3-8B-bf16`                       | `harness`         | preserves trusted hints | Output is very short relative to prompt size (1.5%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=99%                                     |
| `mlx-community/FastVLM-0.5B-bf16`                       | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=77% \| repetitive token=phrase: "2000, 2000, 2000, 2000,..."                                                                       |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `harness`         | preserves trusted hints | Output appears truncated to about 6 tokens. \| nontext prompt burden=99%                                                                                                        |
| `mlx-community/Qwen3.5-27B-4bit`                        | `harness`         | preserves trusted hints | Output appears truncated to about 3 tokens. \| nontext prompt burden=99%                                                                                                        |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| degeneration=character_loop: ' A.' repeated \| repetitive token=phrase: "a. a. a. a...."                                    |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`         | preserves trusted hints | Output appears truncated to about 3 tokens. \| nontext prompt burden=99%                                                                                                        |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98%                                                                                                                                |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=94% \| repetitive token=phrase: "saff saff saff saff..."                                                                           |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99%                                                                                                                                |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99%                                                                                                                                |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `harness`         | preserves trusted hints | Output appears truncated to about 6 tokens. \| nontext prompt burden=99%                                                                                                        |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| degeneration=character_loop: '444' repeated                                                                                 |
| `mlx-community/GLM-4.6V-nvfp4`                          | `harness`         | preserves trusted hints | Output appears truncated to about 8 tokens. \| nontext prompt burden=99%                                                                                                        |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| reasoning leak                                                                                                              |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| degeneration=character_loop: ' 7' repeated \| repetitive token=7                                                            |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99%                                                                                                                                |
| `microsoft/Phi-3.5-vision-instruct`                     | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99%                                                                                                                                |
| `mlx-community/X-Reasoner-7B-8bit`                      | `harness`         | preserves trusted hints | Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (200) \| nontext prompt burden=99% \| degeneration=incomplete_sentence: ends with 'of' |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| degeneration=incomplete_sentence: ends with 'of'                                                                           |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| degeneration=character_loop: ' 1.' repeated \| repetitive token=1.                                                          |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98% \| repetitive token=a\_\_                                                                                                      |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99%                                                                                                                                |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| degeneration=incomplete_sentence: ends with 'as' \| repetitive token=phrase: "as- and lastly as..."                        |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| degeneration=character_loop: '¬' repeated \| repetitive token=phrase: "更 falls 更 falls..."                               |
| `mlx-community/InternVL3-14B-8bit`                      | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| repetitive token=phrase: "2018年，所以 2018年，所以 2018年，所以 201..."                                                    |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| degeneration=character_loop: '555' repeated                                                                                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98%                                                                                                                                |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| formatting=Unknown tags: &lt;fake_token_around_image&gt; \| repetitive token=comm,                                         |
| `mlx-community/pixtral-12b-8bit`                        | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100%                                                                                                                               |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| degeneration=repeated_punctuation: '===...' \| repetitive token====                                                         |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| degeneration=incomplete_sentence: ends with 'to'                                                                            |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98%                                                                                                                                |
| `mlx-community/pixtral-12b-bf16`                        | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100%                                                                                                                               |
| `mlx-community/gemma-4-31b-bf16`                        | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=98%                                                                                                                                |
| `mlx-community/MolmoPoint-8B-fp16`                      | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=100% \| repetitive token=phrase: "there is a black..."                                                                             |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=65% \| degeneration=character_loop: 'ors' repeated                                                                                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% \| reasoning leak                                                                                                              |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                               | Affected Models                              | Issue Draft                                                                                                                              | Evidence Bundle   | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                   | 1: `mlx-community/Kimi-VL-A3B-Thinking-8bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)             | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx`                                                    | Weight/config mismatch during model load              | Weight Mismatch \| phase model_load \| ValueError                                                                                                               | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`       | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx_mlx-model-load-weight-mismatch_001.md)   | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-lm`                                                 | Missing module/import during model load               | Model Error \| phase model_load \| ModuleNotFoundError                                                                                                          | 1: `facebook/pe-av-large`                    | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-lm_mlx-lm-model-load-model_001.md)       | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=803 \| output/prompt=24.91% \| nontext burden=99% \| stop=max_tokens \| hit token cap (200) | 1: `mlx-community/X-Reasoner-7B-8bit`        | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_stop-token_001.md)                   | -                 | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~4 \| prompt=269 \| output/prompt=1.49% \| nontext burden=98% \| stop=completed \| 7 model cluster                                              | 7: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (+6)   | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_model-config-mlx-vlm_prompt-template_001.md) | -                 | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | output/prompt=0.2% \| prompt_tokens=4103, output_tokens=9, output/prompt=0.2% \| prompt=4,103 \| output/prompt=0.22% \| nontext burden=100% \| stop=completed   | 1: `mlx-community/paligemma2-3b-pt-896-4bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm-mlx_long-context_001.md)             | -                 | Full and reduced reruns avoid context collapse.           |

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
- _Key signals:_ Output appears truncated to about 4 tokens.; nontext prompt
  burden=98%
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 4 tok; requested max 200 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 111 tok; requested max 200 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; repetitive
  token=and
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 73 tok; requested max 200 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (1.5%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=99%
- _Tokens:_ prompt 781 tok; estimated text 6 tok; estimated non-text 775 tok;
  generated 12 tok; requested max 200 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=77%; repetitive
  token=phrase: "2000, 2000, 2000, 2000,..."
- _Tokens:_ prompt 26 tok; estimated text 6 tok; estimated non-text 20 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 6 tokens.; nontext prompt
  burden=99%
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 6 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.; nontext prompt
  burden=99%
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 3 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: ' A.' repeated; repetitive token=phrase: "a. a.
  a. a...."
- _Tokens:_ prompt 803 tok; estimated text 6 tok; estimated non-text 797 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


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


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.; nontext prompt
  burden=99%
- _Tokens:_ prompt 807 tok; estimated text 6 tok; estimated non-text 801 tok;
  generated 3 tok; requested max 200 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Tokens:_ prompt 1339 tok; estimated text 6 tok; estimated non-text 1333
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
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


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
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


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Tokens:_ prompt 284 tok; estimated text 6 tok; estimated non-text 278 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: 'T;' repeated
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: '444' repeated
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 8 tokens.; nontext prompt
  burden=99%
- _Tokens:_ prompt 790 tok; estimated text 6 tok; estimated non-text 784 tok;
  generated 8 tok; requested max 200 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: '13-' repeated
- _Tokens:_ prompt 790 tok; estimated text 6 tok; estimated non-text 784 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; reasoning
  leak
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: ' 7' repeated; repetitive token=7
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; hit token cap (200); nontext prompt burden=99%;
  degeneration=incomplete_sentence: ends with 'of'
- _Tokens:_ prompt 803 tok; estimated text 6 tok; estimated non-text 797 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=incomplete_sentence: ends with 'of'
- _Tokens:_ prompt 1340 tok; estimated text 6 tok; estimated non-text 1334
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: ' 1.' repeated; repetitive token=1.
- _Tokens:_ prompt 790 tok; estimated text 6 tok; estimated non-text 784 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 76 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; repetitive
  token=a__
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Tokens:_ prompt 825 tok; estimated text 6 tok; estimated non-text 819 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=incomplete_sentence: ends with 'as'; repetitive token=phrase:
  "as- and lastly as..."
- _Tokens:_ prompt 1964 tok; estimated text 6 tok; estimated non-text 1958
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=character_loop: '¬' repeated; repetitive token=phrase: "更 falls
  更 falls..."
- _Tokens:_ prompt 1340 tok; estimated text 6 tok; estimated non-text 1334
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Tokens:_ prompt 2440 tok; estimated text 6 tok; estimated non-text 2434
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; repetitive
  token=phrase: "2018年，所以 2018年，所以 2018年，所以 201..."
- _Tokens:_ prompt 781 tok; estimated text 6 tok; estimated non-text 775 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: '555' repeated
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
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

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  degeneration=character_loop: 'e_' repeated
- _Tokens:_ prompt 284 tok; estimated text 6 tok; estimated non-text 278 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Tokens:_ prompt 2349 tok; estimated text 6 tok; estimated non-text 2343
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=62%;
  degeneration=repeated_punctuation: '##########...'
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=repeated_punctuation: '===...'; repetitive token====
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=incomplete_sentence: ends with 'to'
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 1201 tok; estimated text 6 tok; estimated non-text 1195
  tok; generated 30 tok; requested max 200 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 1201 tok; estimated text 6 tok; estimated non-text 1195
  tok; generated 48 tok; requested max 200 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Tokens:_ prompt 2349 tok; estimated text 6 tok; estimated non-text 2343
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Tokens:_ prompt 272 tok; estimated text 6 tok; estimated non-text 266 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "there is a black..."
- _Tokens:_ prompt 1203 tok; estimated text 6 tok; estimated non-text 1197
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=65%;
  degeneration=character_loop: 'ors' repeated
- _Tokens:_ prompt 17 tok; estimated text 6 tok; estimated non-text 11 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; reasoning
  leak
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

