<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

_Generated on 2026-06-06 20:51:24 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🏆 A (100/100) | Desc 93 | Keywords 85 | 183.3 tps
- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (100/100) | Desc 93 | Keywords 77 | 105.0 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (100/100) | Desc 93 | Keywords 88 | 66.2 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (100/100) | Desc 79 | Keywords 88 | 63.3 tps
- `mlx-community/InternVL3-14B-8bit`: 🏆 A (100/100) | Desc 93 | Keywords 78 | 32.3 tps

### Watchlist

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | 278.0 tps | harness, long context
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | 27.6 tps | harness, missing sections
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (16/100) | Desc 47 | Keywords 0 | 66.8 tps | harness, missing sections
- `microsoft/Phi-3.5-vision-instruct`: ✅ B (66/100) | Desc 93 | Keywords 52 | 56.3 tps | degeneration, generation loop, harness, text sanity
- `mlx-community/MiniCPM-V-4.6-8bit`: 🏆 A (90/100) | Desc 87 | Keywords 84 | 275.4 tps | harness, reasoning leak, text sanity

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict     | Hint Handling           | Key Evidence                                     |
|-----------------------------------------------------|-------------|-------------------------|--------------------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`     | preserves trusted hints | no flagged signals                               |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `clean`     | preserves trusted hints | nontext prompt burden=88%                        |
| `mlx-community/InternVL3-8B-bf16`                   | `clean`     | preserves trusted hints | nontext prompt burden=84%                        |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`     | preserves trusted hints | nontext prompt burden=88%                        |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`     | preserves trusted hints | nontext prompt burden=88%                        |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`     | preserves trusted hints | nontext prompt burden=84%                        |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`     | preserves trusted hints | no flagged signals                               |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `clean`     | preserves trusted hints | no flagged signals                               |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `clean`     | preserves trusted hints | no flagged signals                               |
| `mlx-community/MolmoPoint-8B-fp16`                  | `clean`     | preserves trusted hints | nontext prompt burden=89%                        |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `token_cap` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=98% |

### `caveat`

| Model                                        | Verdict          | Hint Handling           | Key Evidence                                                                                                                                                                                   |
|----------------------------------------------|------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/LFM2-VL-1.6B-8bit`            | `clean`          | preserves trusted hints | keywords=7                                                                                                                                                                                     |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`   | `clean`          | preserves trusted hints | keywords=5                                                                                                                                                                                     |
| `mlx-community/LFM2.5-VL-1.6B-bf16`          | `clean`          | preserves trusted hints | no flagged signals                                                                                                                                                                             |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `clean`          | preserves trusted hints | nontext prompt burden=71% \| keywords=22                                                                                                                                                       |
| `mlx-community/pixtral-12b-8bit`             | `clean`          | preserves trusted hints | nontext prompt burden=89%                                                                                                                                                                      |
| `mlx-community/gemma-3n-E4B-it-bf16`         | `clean`          | preserves trusted hints | keywords=19                                                                                                                                                                                    |
| `mlx-community/pixtral-12b-bf16`             | `clean`          | preserves trusted hints | nontext prompt burden=89%                                                                                                                                                                      |
| `mlx-community/X-Reasoner-7B-8bit`           | `clean`          | preserves trusted hints | nontext prompt burden=98%                                                                                                                                                                      |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`    | `context_budget` | preserves trusted hints | Output appears truncated to about 4 tokens. \| At long prompt length (16626 tokens), output stayed unusually short (4 tokens; ratio 0.0%). \| output/prompt=0.02% \| nontext prompt burden=98% |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict             | Hint Handling           | Key Evidence                                                                                                                                                                                     |
|---------------------------------------------------------|---------------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | `runtime_failure`   | not evaluated           | model error \| mlx model load model                                                                                                                                                              |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | `model_shortcoming` | preserves trusted hints | missing sections: keywords                                                                                                                                                                       |
| `qnguyen3/nanoLLaVA`                                    | `model_shortcoming` | preserves trusted hints | missing sections: keywords                                                                                                                                                                       |
| `mlx-community/MiniCPM-V-4.6-8bit`                      | `harness`           | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| nontext prompt burden=65% \| keywords=8 \| reasoning leak                                                                    |
| `mlx-community/SmolVLM-Instruct-bf16`                   | `model_shortcoming` | preserves trusted hints | nontext prompt burden=78% \| missing sections: keywords                                                                                                                                          |
| `HuggingFaceTB/SmolVLM-Instruct`                        | `model_shortcoming` | preserves trusted hints | nontext prompt burden=78% \| missing sections: keywords                                                                                                                                          |
| `mlx-community/FastVLM-0.5B-bf16`                       | `model_shortcoming` | preserves trusted hints | missing sections: keywords                                                                                                                                                                       |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | `model_shortcoming` | preserves trusted hints | nontext prompt burden=87% \| missing sections: description, keywords \| formatting=Unknown tags: &lt;end_of_utterance&gt;                                                                        |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | `harness`           | preserves trusted hints | Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=86% \| missing sections: title, description, keywords    |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | `model_shortcoming` | preserves trusted hints | nontext prompt burden=75% \| missing sections: title, description, keywords                                                                                                                      |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| missing sections: title, description, keywords \| repetitive token=phrase: "and 10-18, and 10-18,..."                                                                     |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | `model_shortcoming` | preserves trusted hints | nontext prompt burden=75% \| missing sections: title, description, keywords                                                                                                                      |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | `model_shortcoming` | preserves trusted hints | nontext prompt burden=75% \| missing sections: title \| keywords=21 \| reasoning leak                                                                                                            |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`           | preserves trusted hints | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 46 occurrences). \| nontext prompt burden=86% \| missing sections: description, keywords                              |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=75% \| missing sections: title \| keywords=25                                                                                                       |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`           | preserves trusted hints | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=71% |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=79% \| missing sections: title, description \| keyword duplication=38%                                                                              |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=95% \| missing sections: title, description, keywords \| reasoning leak                                                                             |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=92% \| missing sections: title, description, keywords \| reasoning leak                                                                             |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=89% \| missing sections: title \| keywords=5                                                                                                        |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=95% \| missing sections: title, description, keywords \| reasoning leak                                                                             |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | `model_shortcoming` | preserves trusted hints | nontext prompt burden=75% \| missing sections: title, description, keywords                                                                                                                      |
| `mlx-community/GLM-4.6V-nvfp4`                          | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=95% \| missing sections: title, description \| keywords=43                                                                                          |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| keyword duplication=75% \| repetitive token=phrase: "water sports equipment, water..."                                                                                    |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=78% \| missing sections: title, description, keywords \| repetitive token=phrase: "the boy is wearing..."                                           |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=78% \| missing sections: title, description, keywords                                                                                               |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `model_shortcoming` | preserves trusted hints | nontext prompt burden=75% \| missing sections: title \| reasoning leak                                                                                                                           |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=98% \| missing sections: keywords                                                                                                                   |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=98% \| missing sections: description, keywords                                                                                                      |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=98% \| missing sections: title, description, keywords \| degeneration=incomplete_sentence: ends with 'of'                                           |
| `mlx-community/gemma-4-31b-bf16`                        | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| keywords=105                                                                                                                                                              |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=98% \| missing sections: description, keywords                                                                                                      |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| missing sections: title, description, keywords \| repetitive token=phrase: "water vehicle, water recreatio..."                                                            |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=98% \| missing sections: description, keywords                                                                                                      |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `cutoff_degraded`   | preserves trusted hints | hit token cap (500) \| nontext prompt burden=98%                                                                                                                                                 |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                                                                           | Affected Models                                            | Issue Draft                                                                                                                              | Evidence Bundle   | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                                                                                               | 1: `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)             | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers              | 46 BPE space markers found in decoded text \| prompt=2,490 \| output/prompt=3.21% \| nontext burden=86% \| stop=completed                                                                                                                   | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_encoding_001.md)                     | -                 | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=1,201 \| output/prompt=41.63% \| nontext burden=71% \| stop=max_tokens \| hit token cap (500) \| 2 model cluster | 2: `microsoft/Phi-3.5-vision-instruct` (+1)                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm_stop-token_001.md)                   | -                 | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | output/prompt=0.4% \| prompt=2,588 \| output/prompt=0.39% \| nontext burden=86% \| stop=completed                                                                                                                                           | 1: `mlx-community/llava-v1.6-mistral-7b-8bit`              | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_model-config-mlx-vlm_prompt-template_001.md) | -                 | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | generated_tokens~4 \| prompt_tokens=16626, output_tokens=4, output/prompt=0.0% \| prompt=16,626 \| output/prompt=0.02% \| nontext burden=98% \| stop=completed                                                                              | 1: `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm-mlx_long-context_001.md)             | -                 | Full and reduced reruns avoid context collapse.           |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | token cap \| missing sections \| abrupt tail \| prompt=16,639 \| output/prompt=3.00% \| nontext burden=98% \| stop=max_tokens \| hit token cap (500)                                                                                        | 1: `mlx-community/Qwen3.5-35B-A3B-bf16`                    | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm-mlx_long-context_002.md)             | -                 | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

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


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=7
- _Tokens:_ prompt 659 tok; estimated text 354 tok; estimated non-text 305
  tok; generated 85 tok; requested max 500 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords
- _Tokens:_ prompt 398 tok; estimated text 354 tok; estimated non-text 44 tok;
  generated 95 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords
- _Tokens:_ prompt 398 tok; estimated text 354 tok; estimated non-text 44 tok;
  generated 45 tok; requested max 500 tok; stop reason completed


### `mlx-community/MiniCPM-V-4.6-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; nontext prompt burden=65%; keywords=8; reasoning leak
- _Tokens:_ prompt 1010 tok; estimated text 354 tok; estimated non-text 656
  tok; generated 89 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=5
- _Tokens:_ prompt 507 tok; estimated text 354 tok; estimated non-text 153
  tok; generated 32 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 659 tok; estimated text 354 tok; estimated non-text 305
  tok; generated 165 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=78%; missing sections: keywords
- _Tokens:_ prompt 1606 tok; estimated text 354 tok; estimated non-text 1252
  tok; generated 38 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=78%; missing sections: keywords
- _Tokens:_ prompt 1606 tok; estimated text 354 tok; estimated non-text 1252
  tok; generated 38 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords
- _Tokens:_ prompt 402 tok; estimated text 354 tok; estimated non-text 48 tok;
  generated 162 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 673 tok; estimated text 354 tok; estimated non-text 319
  tok; generated 82 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=88%
- _Tokens:_ prompt 2991 tok; estimated text 354 tok; estimated non-text 2637
  tok; generated 102 tok; requested max 500 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=87%; missing sections: description,
  keywords; formatting=Unknown tags: &lt;end_of_utterance&gt;
- _Tokens:_ prompt 2690 tok; estimated text 354 tok; estimated non-text 2336
  tok; generated 16 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=71%; keywords=22
- _Tokens:_ prompt 1201 tok; estimated text 354 tok; estimated non-text 847
  tok; generated 128 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=84%
- _Tokens:_ prompt 2181 tok; estimated text 354 tok; estimated non-text 1827
  tok; generated 58 tok; requested max 500 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Check chat-template and EOS defaults first; the output shape is
  not matching the requested contract.
- _Key signals:_ Output is very short relative to prompt size (0.4%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=86%; missing sections: title, description, keywords
- _Tokens:_ prompt 2588 tok; estimated text 354 tok; estimated non-text 2234
  tok; generated 10 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=88%
- _Tokens:_ prompt 2992 tok; estimated text 354 tok; estimated non-text 2638
  tok; generated 83 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=88%
- _Tokens:_ prompt 2992 tok; estimated text 354 tok; estimated non-text 2638
  tok; generated 85 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=89%
- _Tokens:_ prompt 3182 tok; estimated text 354 tok; estimated non-text 2828
  tok; generated 83 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=84%
- _Tokens:_ prompt 2181 tok; estimated text 354 tok; estimated non-text 1827
  tok; generated 82 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=75%; missing sections: title,
  description, keywords
- _Tokens:_ prompt 1424 tok; estimated text 354 tok; estimated non-text 1070
  tok; generated 105 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 673 tok; estimated text 354 tok; estimated non-text 319
  tok; generated 88 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; repetitive token=phrase: "and 10-18, and 10-18,..."
- _Tokens:_ prompt 659 tok; estimated text 354 tok; estimated non-text 305
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=75%; missing sections: title,
  description, keywords
- _Tokens:_ prompt 1424 tok; estimated text 354 tok; estimated non-text 1070
  tok; generated 87 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=75%; missing sections: title;
  keywords=21; reasoning leak
- _Tokens:_ prompt 1400 tok; estimated text 354 tok; estimated non-text 1046
  tok; generated 327 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 668 tok; estimated text 354 tok; estimated non-text 314
  tok; generated 94 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=19
- _Tokens:_ prompt 667 tok; estimated text 354 tok; estimated non-text 313
  tok; generated 237 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=89%
- _Tokens:_ prompt 3182 tok; estimated text 354 tok; estimated non-text 2828
  tok; generated 78 tok; requested max 500 tok; stop reason completed


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 46 occurrences).; nontext prompt burden=86%; missing sections:
  description, keywords
- _Tokens:_ prompt 2490 tok; estimated text 354 tok; estimated non-text 2136
  tok; generated 80 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 668 tok; estimated text 354 tok; estimated non-text 314
  tok; generated 98 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title; keywords=25
- _Tokens:_ prompt 1400 tok; estimated text 354 tok; estimated non-text 1046
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; Special control token &lt;|endoftext|&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=71%
- _Tokens:_ prompt 1201 tok; estimated text 354 tok; estimated non-text 847
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=79%; missing
  sections: title, description; keyword duplication=38%
- _Tokens:_ prompt 1707 tok; estimated text 354 tok; estimated non-text 1353
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=95%; missing
  sections: title, description, keywords; reasoning leak
- _Tokens:_ prompt 6520 tok; estimated text 354 tok; estimated non-text 6166
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=92%; missing
  sections: title, description, keywords; reasoning leak
- _Tokens:_ prompt 4496 tok; estimated text 354 tok; estimated non-text 4142
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=89%; missing
  sections: title; keywords=5
- _Tokens:_ prompt 3273 tok; estimated text 354 tok; estimated non-text 2919
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=95%; missing
  sections: title, description, keywords; reasoning leak
- _Tokens:_ prompt 6520 tok; estimated text 354 tok; estimated non-text 6166
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=89%
- _Tokens:_ prompt 3194 tok; estimated text 354 tok; estimated non-text 2840
  tok; generated 89 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=75%; missing sections: title,
  description, keywords
- _Tokens:_ prompt 1424 tok; estimated text 354 tok; estimated non-text 1070
  tok; generated 100 tok; requested max 500 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Tokens:_ prompt 16626 tok; estimated text 354 tok; estimated non-text 16272
  tok; generated 123 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=95%; missing
  sections: title, description; keywords=43
- _Tokens:_ prompt 6520 tok; estimated text 354 tok; estimated non-text 6166
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); keyword duplication=75%; repetitive
  token=phrase: "water sports equipment, water..."
- _Tokens:_ prompt 379 tok; estimated text 354 tok; estimated non-text 25 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=78%; missing
  sections: title, description, keywords; repetitive token=phrase: "the boy is
  wearing..."
- _Tokens:_ prompt 1577 tok; estimated text 354 tok; estimated non-text 1223
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=78%; missing
  sections: title, description, keywords
- _Tokens:_ prompt 1577 tok; estimated text 354 tok; estimated non-text 1223
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 98% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 4 tokens.; At long prompt
  length (16626 tokens), output stayed unusually short (4 tokens; ratio
  0.0%).; output/prompt=0.02%; nontext prompt burden=98%
- _Tokens:_ prompt 16626 tok; estimated text 354 tok; estimated non-text 16272
  tok; generated 4 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=75%; missing sections: title; reasoning
  leak
- _Tokens:_ prompt 1400 tok; estimated text 354 tok; estimated non-text 1046
  tok; generated 281 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=98%; missing
  sections: keywords
- _Tokens:_ prompt 16639 tok; estimated text 354 tok; estimated non-text 16285
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=98%; missing
  sections: description, keywords
- _Tokens:_ prompt 16639 tok; estimated text 354 tok; estimated non-text 16285
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); nontext prompt burden=98%
- _Tokens:_ prompt 16639 tok; estimated text 354 tok; estimated non-text 16285
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords; degeneration=incomplete_sentence:
  ends with 'of'
- _Tokens:_ prompt 16639 tok; estimated text 354 tok; estimated non-text 16285
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); keywords=105
- _Tokens:_ prompt 661 tok; estimated text 354 tok; estimated non-text 307
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=98%; missing
  sections: description, keywords
- _Tokens:_ prompt 16639 tok; estimated text 354 tok; estimated non-text 16285
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; repetitive token=phrase: "water vehicle, water recreatio..."
- _Tokens:_ prompt 380 tok; estimated text 354 tok; estimated non-text 26 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=98%; missing
  sections: description, keywords
- _Tokens:_ prompt 16639 tok; estimated text 354 tok; estimated non-text 16285
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); nontext prompt burden=98%
- _Tokens:_ prompt 16639 tok; estimated text 354 tok; estimated non-text 16285
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

