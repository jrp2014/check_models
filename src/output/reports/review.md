# Automated Review Digest

_Generated on 2026-05-02 00:10:46 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/gemma-3-27b-it-qat-4bit`: 🏆 A (85/100) | Desc 82 | Keywords 92 | 27.1 tps
- `mlx-community/Qwen3.5-35B-A3B-bf16`: ✅ B (80/100) | Desc 81 | Keywords 90 | 55.4 tps
- `mlx-community/Molmo-7B-D-0924-8bit`: ✅ B (80/100) | Desc 84 | Keywords 0 | 51.8 tps
- `mlx-community/Molmo-7B-D-0924-bf16`: ✅ B (80/100) | Desc 84 | Keywords 0 | 30.5 tps
- `mlx-community/gemma-3-27b-it-qat-8bit`: ✅ B (80/100) | Desc 82 | Keywords 94 | 17.6 tps

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 40 | Keywords 0 | 31.4 tps | harness
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (5/100) | Desc 22 | Keywords 0 | 83.9 tps | harness, long context
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (5/100) | Desc 23 | Keywords 0 | 92.7 tps | harness
- `mlx-community/gemma-4-31b-bf16`: ❌ F (5/100) | Desc 23 | Keywords 0 | 9.2 tps | harness
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (16/100) | Desc 48 | Keywords 0 | 209.5 tps | harness, long context

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                                               | Verdict     | Hint Handling           | Key Evidence                                                                                     |
|-----------------------------------------------------|-------------|-------------------------|--------------------------------------------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`                  | `clean`     | preserves trusted hints | nontext prompt burden=80%                                                                        |
| `qnguyen3/nanoLLaVA`                                | `clean`     | preserves trusted hints | nontext prompt burden=80%                                                                        |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                 | `clean`     | preserves trusted hints | nontext prompt burden=95%                                                                        |
| `mlx-community/LFM2-VL-1.6B-8bit`                   | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/FastVLM-0.5B-bf16`                   | `clean`     | preserves trusted hints | nontext prompt burden=83%                                                                        |
| `mlx-community/Idefics3-8B-Llama3-bf16`             | `clean`     | preserves trusted hints | nontext prompt burden=100% \| formatting=Unknown tags: &lt;end_of_utterance&gt;                  |
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`   | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/llava-v1.6-mistral-7b-8bit`          | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `HuggingFaceTB/SmolVLM-Instruct`                    | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/SmolVLM-Instruct-bf16`               | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/gemma-3n-E4B-it-bf16`                | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/InternVL3-14B-8bit`                  | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`             | `clean`     | preserves trusted hints | nontext prompt burden=100% \| reasoning leak                                                     |
| `mlx-community/pixtral-12b-8bit`                    | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`     | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/gemma-4-31b-it-4bit`                 | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/pixtral-12b-bf16`                    | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/InternVL3-8B-bf16`                   | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/gemma-3-27b-it-qat-4bit`             | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/X-Reasoner-7B-8bit`                  | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`  | `clean`     | preserves trusted hints | nontext prompt burden=71%                                                                        |
| `mlx-community/Molmo-7B-D-0924-8bit`                | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `mlx-community/gemma-3-27b-it-qat-8bit`             | `clean`     | preserves trusted hints | nontext prompt burden=99%                                                                        |
| `mlx-community/Molmo-7B-D-0924-bf16`                | `clean`     | preserves trusted hints | nontext prompt burden=100%                                                                       |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`          | `clean`     | preserves trusted hints | nontext prompt burden=73%                                                                        |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`      | `clean`     | preserves trusted hints | nontext prompt burden=100% \| reasoning leak                                                     |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 | `token_cap` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100% \| degeneration=repeated_punctuation: ':**...' |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                | `token_cap` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                |

### `caveat`

| Model                                     | Verdict          | Hint Handling           | Key Evidence                                                                                                                                                                                                                                                        |
|-------------------------------------------|------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/paligemma2-3b-pt-896-4bit` | `context_budget` | preserves trusted hints | Output appears truncated to about 3 tokens. \| At long prompt length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%). \| output/prompt=0.07% \| nontext prompt burden=100%                                                                      |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | preserves trusted hints | Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues. \| At long prompt length (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%). \| output/prompt=0.08% \| nontext prompt burden=100% |

### `needs_triage`

- None.

### `avoid`

| Model                                                   | Verdict           | Hint Handling           | Key Evidence                                                                                                                                                                                     |
|---------------------------------------------------------|-------------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | `runtime_failure` | not evaluated           | model error \| mlx model load model                                                                                                                                                              |
| `mlx-community/MolmoPoint-8B-fp16`                      | `runtime_failure` | not evaluated           | processor error \| model config processor load processor                                                                                                                                         |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | `runtime_failure` | not evaluated           | model error \| huggingface hub model load model \| hub connectivity                                                                                                                              |
| `mlx-community/gemma-3n-E2B-4bit`                       | `harness`         | preserves trusted hints | Output appears truncated to about 4 tokens. \| nontext prompt burden=98%                                                                                                                         |
| `mlx-community/gemma-4-31b-bf16`                        | `harness`         | preserves trusted hints | Output appears truncated to about 5 tokens. \| nontext prompt burden=98%                                                                                                                         |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness`         | preserves trusted hints | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 139 occurrences). \| nontext prompt burden=100%                                                                       |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness`         | preserves trusted hints | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=99% |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100% \| reasoning leak \| degeneration=incomplete_sentence: ends with 'of'                                                                          |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `harness`         | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `harness`         | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          |
| `mlx-community/GLM-4.6V-nvfp4`                          | `harness`         | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |
| `mlx-community/Qwen3.5-27B-4bit`                        | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |
| `mlx-community/Qwen3.6-27B-mxfp8`                       | `cutoff_degraded` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100%                                                                                                                                                |

## Maintainer Queue

Owner-grouped escalations only; clean recommended models stay in user buckets.

### `huggingface-hub`

| Model                                      | Verdict           | Evidence                                                            | Next Action                                                                                                                    |
|--------------------------------------------|-------------------|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `runtime_failure` | model error \| huggingface hub model load model \| hub connectivity | Check whether Hugging Face was reachable; this may be a transient Hub/network outage or disconnect rather than a model defect. |

### `mlx`

| Model                                     | Verdict           | Evidence                                                                                                                                                                                                                                                            | Next Action                                                                                                         |
|-------------------------------------------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `runtime_failure` | model error \| mlx model load model                                                                                                                                                                                                                                 | Inspect KV/cache behavior, memory pressure, and long-context execution.                                             |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `context_budget`  | Output appears truncated to about 3 tokens. \| At long prompt length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%). \| output/prompt=0.07% \| nontext prompt burden=100%                                                                      | Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget`  | Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues. \| At long prompt length (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%). \| output/prompt=0.08% \| nontext prompt burden=100% | Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load. |

### `mlx-vlm`

| Model                                                   | Verdict   | Evidence                                                                                                                                                                                         | Next Action                                                                         |
|---------------------------------------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 139 occurrences). \| nontext prompt burden=100%                                                                       | Inspect decode cleanup; tokenizer markers are leaking into user-facing text.        |
| `microsoft/Phi-3.5-vision-instruct`                     | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=99% | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-nvfp4`                          | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak                                                          | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model                                              | Verdict           | Evidence                                                                                                                | Next Action                                                    |
|----------------------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| `mlx-community/Idefics3-8B-Llama3-bf16`            | `clean`           | nontext prompt burden=100% \| formatting=Unknown tags: &lt;end_of_utterance&gt;                                         | Treat as a model-quality limitation for this prompt and image. |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`            | `clean`           | nontext prompt burden=100% \| reasoning leak                                                                            | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`    | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100% \| reasoning leak \| degeneration=incomplete_sentence: ends with 'of' | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`     | `clean`           | nontext prompt burden=100% \| reasoning leak                                                                            | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                | `token_cap`       | hit token cap (500) \| nontext prompt burden=100% \| degeneration=repeated_punctuation: ':**...'                        | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-35B-A3B-4bit`               | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-35B-A3B-6bit`               | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-27B-4bit`                   | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-27B-mxfp8`                  | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.6-27B-mxfp8`                  | `cutoff_degraded` | hit token cap (500) \| nontext prompt burden=100%                                                                       | Treat as a model-quality limitation for this prompt and image. |

### `model-config`

| Model                              | Verdict           | Evidence                                                                 | Next Action                                                 |
|------------------------------------|-------------------|--------------------------------------------------------------------------|-------------------------------------------------------------|
| `mlx-community/MolmoPoint-8B-fp16` | `runtime_failure` | processor error \| model config processor load processor                 | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/gemma-3n-E2B-4bit`  | `harness`         | Output appears truncated to about 4 tokens. \| nontext prompt burden=98% | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/gemma-4-31b-bf16`   | `harness`         | Output appears truncated to about 5 tokens. \| nontext prompt burden=98% | Inspect model repo config, chat template, and EOS settings. |

## Model Verdicts

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Model
  Error`; diagnostic code `MLX_MODEL_LOAD_MODEL`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ model error; mlx model load model
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `mlx-community/MolmoPoint-8B-fp16`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `model-config`; reported package `model-config`;
  failure stage `Processor Error`; diagnostic code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ processor error; model config processor load processor
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Owner:_ likely owner `huggingface-hub`; reported package `huggingface-hub`;
  failure stage `Model Error`; diagnostic code
  `HUGGINGFACE_HUB_MODEL_LOAD_MODEL`
- _Next step:_ Check whether Hugging Face was reachable; this may be a
  transient Hub/network outage or disconnect rather than a model defect.
- _Key signals:_ model error; huggingface hub model load model; hub
  connectivity
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


### `mlx-community/gemma-3n-E2B-4bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 4 tokens.; nontext prompt
  burden=98%
- _Tokens:_ prompt 264 tok; estimated text 4 tok; estimated non-text 260 tok;
  generated 4 tok; requested max 500 tok; stop reason completed


### `mlx-community/nanoLLaVA-1.5-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=80%
- _Tokens:_ prompt 20 tok; estimated text 4 tok; estimated non-text 16 tok;
  generated 124 tok; requested max 500 tok; stop reason completed


### `qnguyen3/nanoLLaVA`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=80%
- _Tokens:_ prompt 20 tok; estimated text 4 tok; estimated non-text 16 tok;
  generated 51 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2.5-VL-1.6B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=95%
- _Tokens:_ prompt 85 tok; estimated text 4 tok; estimated non-text 81 tok;
  generated 117 tok; requested max 500 tok; stop reason completed


### `mlx-community/LFM2-VL-1.6B-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Tokens:_ prompt 275 tok; estimated text 4 tok; estimated non-text 271 tok;
  generated 292 tok; requested max 500 tok; stop reason completed


### `mlx-community/Phi-3.5-vision-instruct-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Tokens:_ prompt 768 tok; estimated text 4 tok; estimated non-text 764 tok;
  generated 53 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 3 tokens.; At long prompt
  length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%).;
  output/prompt=0.07%; nontext prompt burden=100%
- _Tokens:_ prompt 4101 tok; estimated text 4 tok; estimated non-text 4097
  tok; generated 3 tok; requested max 500 tok; stop reason completed


### `mlx-community/FastVLM-0.5B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=83%
- _Tokens:_ prompt 24 tok; estimated text 4 tok; estimated non-text 20 tok;
  generated 445 tok; requested max 500 tok; stop reason completed


### `mlx-community/Idefics3-8B-Llama3-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%; formatting=Unknown tags:
  &lt;end_of_utterance&gt;
- _Tokens:_ prompt 2326 tok; estimated text 4 tok; estimated non-text 2322
  tok; generated 48 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-26b-a4b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Tokens:_ prompt 278 tok; estimated text 4 tok; estimated non-text 274 tok;
  generated 267 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-bf16`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 5 tokens.; nontext prompt
  burden=98%
- _Tokens:_ prompt 266 tok; estimated text 4 tok; estimated non-text 262 tok;
  generated 5 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 2629 tok; estimated text 4 tok; estimated non-text 2625
  tok; generated 355 tok; requested max 500 tok; stop reason completed


### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 2174 tok; estimated text 4 tok; estimated non-text 2170
  tok; generated 74 tok; requested max 500 tok; stop reason completed


### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 1195 tok; estimated text 4 tok; estimated non-text 1191
  tok; generated 420 tok; requested max 500 tok; stop reason completed


### `mlx-community/SmolVLM-Instruct-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 1195 tok; estimated text 4 tok; estimated non-text 1191
  tok; generated 420 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 1029 tok; estimated text 4 tok; estimated non-text 1025
  tok; generated 135 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3n-E4B-it-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Tokens:_ prompt 272 tok; estimated text 4 tok; estimated non-text 268 tok;
  generated 237 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-14B-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 1804 tok; estimated text 4 tok; estimated non-text 1800
  tok; generated 125 tok; requested max 500 tok; stop reason completed


### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%; reasoning leak
- _Tokens:_ prompt 1011 tok; estimated text 4 tok; estimated non-text 1007
  tok; generated 357 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 2887 tok; estimated text 4 tok; estimated non-text 2883
  tok; generated 167 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 1029 tok; estimated text 4 tok; estimated non-text 1025
  tok; generated 118 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-4-31b-it-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Tokens:_ prompt 278 tok; estimated text 4 tok; estimated non-text 274 tok;
  generated 158 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 2630 tok; estimated text 4 tok; estimated non-text 2626
  tok; generated 311 tok; requested max 500 tok; stop reason completed


### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 2630 tok; estimated text 4 tok; estimated non-text 2626
  tok; generated 344 tok; requested max 500 tok; stop reason completed


### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 139 occurrences).; nontext prompt burden=100%
- _Tokens:_ prompt 2097 tok; estimated text 4 tok; estimated non-text 2093
  tok; generated 172 tok; requested max 500 tok; stop reason completed


### `microsoft/Phi-3.5-vision-instruct`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; Special control token &lt;|endoftext|&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=99%
- _Tokens:_ prompt 768 tok; estimated text 4 tok; estimated non-text 764 tok;
  generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/pixtral-12b-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 2887 tok; estimated text 4 tok; estimated non-text 2883
  tok; generated 166 tok; requested max 500 tok; stop reason completed


### `mlx-community/InternVL3-8B-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 1804 tok; estimated text 4 tok; estimated non-text 1800
  tok; generated 344 tok; requested max 500 tok; stop reason completed


### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); nontext prompt burden=100%
- _Tokens:_ prompt 1293 tok; estimated text 4 tok; estimated non-text 1289
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-4bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Tokens:_ prompt 273 tok; estimated text 4 tok; estimated non-text 269 tok;
  generated 320 tok; requested max 500 tok; stop reason completed


### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); nontext prompt burden=100%; reasoning
  leak; degeneration=incomplete_sentence: ends with 'of'
- _Tokens:_ prompt 2979 tok; estimated text 4 tok; estimated non-text 2975
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=100%; reasoning leak
- _Tokens:_ prompt 6091 tok; estimated text 4 tok; estimated non-text 6087
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-Flash-6bit`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=100%; reasoning leak
- _Tokens:_ prompt 6091 tok; estimated text 4 tok; estimated non-text 6087
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 1029 tok; estimated text 4 tok; estimated non-text 1025
  tok; generated 100 tok; requested max 500 tok; stop reason completed


### `mlx-community/X-Reasoner-7B-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 16299 tok; estimated text 4 tok; estimated non-text 16295
  tok; generated 219 tok; requested max 500 tok; stop reason completed


### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=71%
- _Tokens:_ prompt 14 tok; estimated text 4 tok; estimated non-text 10 tok;
  generated 421 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 1207 tok; estimated text 4 tok; estimated non-text 1203
  tok; generated 205 tok; requested max 500 tok; stop reason completed


### `mlx-community/gemma-3-27b-it-qat-8bit`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Tokens:_ prompt 273 tok; estimated text 4 tok; estimated non-text 269 tok;
  generated 381 tok; requested max 500 tok; stop reason completed


### `mlx-community/Molmo-7B-D-0924-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 1207 tok; estimated text 4 tok; estimated non-text 1203
  tok; generated 205 tok; requested max 500 tok; stop reason completed


### `mlx-community/GLM-4.6V-nvfp4`

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=100%; reasoning leak
- _Tokens:_ prompt 6091 tok; estimated text 4 tok; estimated non-text 6087
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Tokens:_ prompt 15 tok; estimated text 4 tok; estimated non-text 11 tok;
  generated 276 tok; requested max 500 tok; stop reason completed


### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%; reasoning leak
- _Tokens:_ prompt 1011 tok; estimated text 4 tok; estimated non-text 1007
  tok; generated 277 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.
- _Key signals:_ Output is very short relative to prompt size (0.1%),
  suggesting possible early-stop or prompt-handling issues.; At long prompt
  length (16299 tokens), output stayed unusually short (13 tokens; ratio
  0.1%).; output/prompt=0.08%; nontext prompt burden=100%
- _Tokens:_ prompt 16299 tok; estimated text 4 tok; estimated non-text 16295
  tok; generated 13 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); nontext prompt burden=100%;
  degeneration=repeated_punctuation: ':**...'
- _Tokens:_ prompt 16290 tok; estimated text 4 tok; estimated non-text 16286
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); nontext prompt burden=100%
- _Tokens:_ prompt 16290 tok; estimated text 4 tok; estimated non-text 16286
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); nontext prompt burden=100%
- _Tokens:_ prompt 16290 tok; estimated text 4 tok; estimated non-text 16286
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); nontext prompt burden=100%
- _Tokens:_ prompt 16290 tok; estimated text 4 tok; estimated non-text 16286
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-4bit`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); nontext prompt burden=100%
- _Tokens:_ prompt 16290 tok; estimated text 4 tok; estimated non-text 16286
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.5-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); nontext prompt burden=100%
- _Tokens:_ prompt 16290 tok; estimated text 4 tok; estimated non-text 16286
  tok; generated 500 tok; requested max 500 tok; stop reason completed


### `mlx-community/Qwen3.6-27B-mxfp8`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); nontext prompt burden=100%
- _Tokens:_ prompt 16290 tok; estimated text 4 tok; estimated non-text 16286
  tok; generated 500 tok; requested max 500 tok; stop reason completed

