<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

_Generated on 2026-06-27 23:58:03 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Companion artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `Qwen/Qwen3-VL-2B-Instruct`: caption-review candidate | 132.9 tps
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: caption-review candidate | 132.6 tps
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: caption-review candidate | 129.3 tps
- `mlx-community/Qwen3.5-35B-A3B-4bit`: caption-review candidate | 120.2 tps
- `mlx-community/Qwen3.5-9B-MLX-4bit`: caption-review candidate | 98.5 tps

### Watchlist

- `mlx-community/paligemma2-3b-pt-896-4bit`: caption-review candidate | 71.9 tps | harness, long context
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: caption-review candidate | 32.8 tps | harness
- `microsoft/Phi-3.5-vision-instruct`: caption-review candidate | 56.3 tps | generation loop, harness, text sanity
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: caption-review candidate | 44.1 tps | harness, reasoning leak, text sanity
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: caption-review candidate | 102.0 tps | harness, text sanity

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict     | Hint Handling   | Key Evidence                                      |
|-----------------------------------------------------|-------------|-----------------|---------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                  | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/FastVLM-0.5B-bf16`                   | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/LFM2-VL-1.6B-8bit`                   | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/MiniCPM-V-4.6-8bit`                  | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`           | `clean`     | not evaluated   | no flagged signals                                |
| `qnguyen3/nanoLLaVA`                                | `clean`     | not evaluated   | no flagged signals                                |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/SmolVLM-Instruct-bf16`               | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                | `clean`     | not evaluated   | no flagged signals                                |
| `Qwen/Qwen3-VL-2B-Instruct`                         | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`           | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/GLM-4.6V-Flash-6bit`                 | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `clean`     | not evaluated   | formatting=Unknown tags: &lt;end_of_utterance&gt; |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`          | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/X-Reasoner-7B-8bit`                  | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`     | not evaluated   | no flagged signals                                |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `token_cap` | not evaluated   | hit token cap (200) \| reasoning leak             |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen3.5-27B-4bit`                    | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen3.6-27B-mxfp8`                   | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`     | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/InternVL3-8B-bf16`                   | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-3n-E4B-it-bf16`                | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/llava-v1.6-mistral-7b-8bit`          | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`      | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Molmo-7B-D-0924-8bit`                | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Molmo-7B-D-0924-bf16`                | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/GLM-4.6V-nvfp4`                      | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`     | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/gemma-4-31b-bf16`                    | `clean`     | not evaluated   | no flagged signals                                |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `clean`     | not evaluated   | no flagged signals                                |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | `clean`     | not evaluated   | no flagged signals                                |

### `caveat`

| Model                                     | Verdict          | Hint Handling   | Key Evidence                                                                                                                                                                                                               |
|-------------------------------------------|------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/paligemma2-3b-pt-896-4bit` | `context_budget` | not evaluated   | Output appears truncated to about 3 tokens. \| At long prompt length (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%; weak text signal truncated). \| output/prompt=0.07% \| nontext prompt burden=100% |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict           | Hint Handling   | Key Evidence                                                                                                                                                        |
|---------------------------------------------------------|-------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | `runtime_failure` | not evaluated   | weight mismatch \| mlx model load weight mismatch                                                                                                                   |
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure` | not evaluated   | model error \| mlx vlm model load model                                                                                                                             |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | `harness`         | not evaluated   | Special control token &lt;/think&gt; appeared in generated text.                                                                                                    |
| `mlx-community/gemma-3n-E2B-4bit`                       | `cutoff_degraded` | not evaluated   | hit token cap (200) \| repetitive token=phrase: "have this image. have..."                                                                                          |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `cutoff_degraded` | not evaluated   | hit token cap (200) \| reasoning leak                                                                                                                               |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`         | not evaluated   | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 76 occurrences).                                                                         |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`         | not evaluated   | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (200) |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `harness`         | not evaluated   | Special control token &lt;\|end\|&gt; appeared in generated text. \| reasoning leak                                                                                 |
| `mlx-community/pixtral-12b-8bit`                        | `cutoff_degraded` | not evaluated   | hit token cap (200)                                                                                                                                                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded` | not evaluated   | hit token cap (200) \| degeneration=incomplete_sentence: ends with 'a'                                                                                              |
| `mlx-community/pixtral-12b-bf16`                        | `cutoff_degraded` | not evaluated   | hit token cap (200)                                                                                                                                                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | `cutoff_degraded` | not evaluated   | hit token cap (200) \| reasoning leak                                                                                                                               |

## Maintainer Escalations

Focused upstream issue drafts are queued in [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).

<!-- markdownlint-disable MD060 -->

| Target                                         | Problem                                                                                          | Evidence Snapshot                                                                                                                                                                                                                         | Affected Models                                            | Issue Draft                                                                                                                            | Evidence Bundle   | Fixed When                                                |
|------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------|
| `mlx`                                          | Weight/config mismatch during model load                                                         | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                                                                         | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-weight-mismatch_001.md) | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | mlx-vlm: Model load / model error: property 'eos_token_id' of 'ModelConfig' object has no setter | Model Error \| phase model_load \| AttributeError                                                                                                                                                                                         | 1: `mlx-community/MolmoPoint-8B-fp16`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_mlx-vlm-model-load-model_001.md)   | -                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | Tokenizer decode leaked BPE/byte markers                                                         | 76 BPE space markers found in decoded text \| prompt=417 \| output/prompt=21.58% \| nontext burden=99% \| stop=completed                                                                                                                  | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm_encoding_001.md)                   | -                 | No BPE/byte markers in output.                            |
| `mlx-vlm`                                      | Stop/control tokens leaked into generated text                                                   | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=770 \| output/prompt=25.97% \| nontext burden=99% \| stop=max_tokens \| hit token cap (200) \| 3 model cluster | 3: `microsoft/Phi-3.5-vision-instruct` (+2)                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_stop-token_001.md)                 | -                 | No leaked stop/control tokens.                            |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short                                            | generated_tokens~3 \| prompt_tokens=4103, output_tokens=3, output/prompt=0.1%, weak text=truncated \| prompt=4,103 \| output/prompt=0.07% \| nontext burden=100% \| stop=completed                                                        | 1: `mlx-community/paligemma2-3b-pt-896-4bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm-mlx_long-context_001.md)           | -                 | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

## Model Verdicts

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


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx-vlm`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect prompt-template, stop-token, and decode post-processing
  behavior.
- _Key signals:_ model error; mlx vlm model load model
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 80 tok; estimated text 6 tok; estimated non-text 74 tok;
  generated 10 tok; requested max 200 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 26 tok; estimated text 6 tok; estimated non-text 20 tok;
  generated 15 tok; requested max 200 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 10 tok; requested max 200 tok; stop reason completed


### `mlx-community/MiniCPM-V-4.6-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 228 tok; estimated text 6 tok; estimated non-text 222 tok;
  generated 22 tok; requested max 200 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 81 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 52 tok; requested max 200 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 35 tok; requested max 200 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 24 tok; requested max 200 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 19 tok; requested max 200 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 42 tok; requested max 200 tok; stop reason completed


### `Qwen/Qwen3-VL-2B-Instruct`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 96 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 96 tok; requested max 200 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 949 tok; estimated text 6 tok; estimated non-text 943 tok;
  generated 99 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 63 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 71 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 58 tok; requested max 200 tok; stop reason completed


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


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 69 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 25 tok; requested max 200 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ formatting=Unknown tags: &lt;end_of_utterance&gt;
- _Tokens:_ prompt 2327 tok; estimated text 6 tok; estimated non-text 2321
  tok; generated 23 tok; requested max 200 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 950 tok; estimated text 6 tok; estimated non-text 944 tok;
  generated 52 tok; requested max 200 tok; stop reason completed


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 97 tok; estimated text 6 tok; estimated non-text 91 tok;
  generated 187 tok; requested max 200 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 65 tok; requested max 200 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 950 tok; estimated text 6 tok; estimated non-text 944 tok;
  generated 68 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.
- _Tokens:_ prompt 317 tok; estimated text 6 tok; estimated non-text 311 tok;
  generated 196 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); repetitive token=phrase: "have this
  image. have..."
- _Tokens:_ prompt 266 tok; estimated text 6 tok; estimated non-text 260 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); reasoning leak
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 40 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 71 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 49 tok; requested max 200 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); reasoning leak
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 76 occurrences).
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 90 tok; requested max 200 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 16 tok; requested max 200 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 3341 tok; estimated text 6 tok; estimated non-text 3335
  tok; generated 48 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 124 tok; requested max 200 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; Special control token &lt;|endoftext|&gt; appeared in generated
  text.; hit token cap (200)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 25 tok; requested max 200 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 2356 tok; estimated text 6 tok; estimated non-text 2350
  tok; generated 55 tok; requested max 200 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 3341 tok; estimated text 6 tok; estimated non-text 3335
  tok; generated 15 tok; requested max 200 tok; stop reason completed


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; reasoning leak
- _Tokens:_ prompt 1330 tok; estimated text 6 tok; estimated non-text 1324
  tok; generated 174 tok; requested max 200 tok; stop reason completed


### `mlx-community/diffusiongemma-26B-A4B-it-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 26 tok; requested max 200 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 101 tok; requested max 200 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 55 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 70 tok; requested max 200 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200)
- _Tokens:_ prompt 1239 tok; estimated text 6 tok; estimated non-text 1233
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 55 tok; requested max 200 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 91 tok; requested max 200 tok; stop reason completed


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 71 tok; requested max 200 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 55 tok; requested max 200 tok; stop reason completed


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 119 tok; requested max 200 tok; stop reason completed


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); degeneration=incomplete_sentence: ends
  with 'a'
- _Tokens:_ prompt 439 tok; estimated text 6 tok; estimated non-text 433 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 36 tok; requested max 200 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200)
- _Tokens:_ prompt 1239 tok; estimated text 6 tok; estimated non-text 1233
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ no flagged signals
- _Tokens:_ prompt 17 tok; estimated text 6 tok; estimated non-text 11 tok;
  generated 77 tok; requested max 200 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
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

