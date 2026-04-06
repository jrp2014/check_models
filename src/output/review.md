# Automated Review Digest

_Generated on 2026-04-06 23:09:06 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [check_models.log](check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/nanoLLaVA-1.5-4bit`: 🏆 A (80/100) | 380.0 tps
- `mlx-community/LFM2.5-VL-1.6B-bf16`: 🏆 A (80/100) | 190.4 tps
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: 🏆 A (80/100) | 34.2 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: ✅ B (79/100) | 59.2 tps
- `mlx-community/InternVL3-14B-8bit`: ✅ B (78/100) | 32.3 tps

### Watchlist

- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (0/100) | 76.0 tps | harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (0/100) | 10.8 tps | harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | 31.1 tps | harness
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) | 239.8 tps | harness, long context
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (17/100) | 62.1 tps | harness, long context

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/LFM2-VL-1.6B-8bit` | `clean` | preserves trusted hints | nontext prompt burden=99% |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `clean` | preserves trusted hints | nontext prompt burden=99% |
| `mlx-community/nanoLLaVA-1.5-4bit` | `clean` | preserves trusted hints | nontext prompt burden=80% |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `clean` | preserves trusted hints | nontext prompt burden=95% |
| `mlx-community/FastVLM-0.5B-bf16` | `clean` | preserves trusted hints | nontext prompt burden=83% |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/InternVL3-14B-8bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `clean` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=96% |
| `mlx-community/gemma-3n-E4B-it-bf16` | `clean` | preserves trusted hints | nontext prompt burden=99% |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/pixtral-12b-8bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `clean` | preserves trusted hints | nontext prompt burden=99% |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `clean` | preserves trusted hints | nontext prompt burden=99% |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `clean` | preserves trusted hints | nontext prompt burden=71% |
| `mlx-community/InternVL3-8B-bf16` | `clean` | preserves trusted hints | nontext prompt burden=100% \| repetitive token=phrase: "rencontre rencontre rencontre ..." |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `clean` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100% |
| `mlx-community/Molmo-7B-D-0924-8bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/X-Reasoner-7B-8bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `clean` | preserves trusted hints | nontext prompt burden=73% |
| `mlx-community/pixtral-12b-bf16` | `clean` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100% |

### `caveat`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `context_budget` | preserves trusted hints | Output is very short relative to prompt size (0.3%), suggesting possible early-stop or prompt-handling issues. \| At long prompt length (4101 tokens), output stayed unusually short (12 tokens; ratio 0.3%). \| output/prompt=0.29% \| nontext prompt burden=100% |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | preserves trusted hints | Output appears truncated to about 4 tokens. \| At long prompt length (16280 tokens), output stayed unusually short (4 tokens; ratio 0.0%). \| output/prompt=0.02% \| nontext prompt burden=100% |

### `avoid`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | not evaluated | processor error \| model config processor load processor |
| `mlx-community/gemma-3n-E2B-4bit` | `harness` | preserves trusted hints | Output appears truncated to about 2 tokens. \| nontext prompt burden=98% |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | preserves trusted hints | Output appears truncated to about 2 tokens. \| nontext prompt burden=100% |
| `qnguyen3/nanoLLaVA` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=2500.00% \| nontext prompt burden=80% \| repetitive token=Baz |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=32.07% \| nontext prompt burden=100% \| repetitive token=unt |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=32.07% \| nontext prompt burden=100% \| repetitive token=unt |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=47.76% \| nontext prompt burden=100% \| degeneration=character_loop: ' 1.' repeated |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=47.76% \| nontext prompt burden=100% \| repetitive token=phrase: "1. these information is..." |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=47.76% \| nontext prompt burden=100% \| degeneration=character_loop: '. 0' repeated |
| `microsoft/Phi-3.5-vision-instruct` | `harness` | preserves trusted hints | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=99% |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | preserves trusted hints | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 127 occurrences). \| nontext prompt burden=100% |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `harness` | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `harness` | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| degeneration=incomplete_sentence: ends with 'a' |
| `mlx-community/GLM-4.6V-Flash-6bit` | `harness` | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=11.74% \| nontext prompt burden=100% \| reasoning leak |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | preserves trusted hints | At long prompt length (16269 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=48.59% \| nontext prompt burden=100% \| repetitive token=phrase: "the light fixtures are..." |
| `mlx-community/GLM-4.6V-nvfp4` | `harness` | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak |
| `mlx-community/Molmo-7B-D-0924-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=41.70% \| nontext prompt burden=100% |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% \| degeneration=incomplete_sentence: ends with 's' |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% \| degeneration=incomplete_sentence: ends with 'of' |
| `mlx-community/Qwen3.5-27B-4bit` | `harness` | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `mlx`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `context_budget` | Output is very short relative to prompt size (0.3%), suggesting possible early-stop or prompt-handling issues. \| At long prompt length (4101 tokens), output stayed unusually short (12 tokens; ratio 0.3%). \| output/prompt=0.29% \| nontext prompt burden=100% | Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load. |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | At long prompt length (16269 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | Output appears truncated to about 4 tokens. \| At long prompt length (16280 tokens), output stayed unusually short (4 tokens; ratio 0.0%). \| output/prompt=0.02% \| nontext prompt burden=100% | Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load. |

### `mlx-vlm`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `microsoft/Phi-3.5-vision-instruct` | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=99% | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 127 occurrences). \| nontext prompt burden=100% | Inspect decode cleanup; tokenizer markers are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| degeneration=incomplete_sentence: ends with 'a' | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-Flash-6bit` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-nvfp4` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Qwen3.5-27B-4bit` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/LFM2-VL-1.6B-8bit` | `clean` | nontext prompt burden=99% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `clean` | nontext prompt burden=99% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/nanoLLaVA-1.5-4bit` | `clean` | nontext prompt burden=80% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `clean` | nontext prompt burden=95% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/FastVLM-0.5B-bf16` | `clean` | nontext prompt burden=83% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/InternVL3-14B-8bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `clean` | hit token cap (500) \| nontext prompt burden=96% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/gemma-3n-E4B-it-bf16` | `clean` | nontext prompt burden=99% | Treat as a model-quality limitation for this prompt and image. |
| `qnguyen3/nanoLLaVA` | `cutoff` | hit token cap (500) \| output/prompt=2500.00% \| nontext prompt burden=80% \| repetitive token=Baz | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2500.00%. |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | hit token cap (500) \| output/prompt=32.07% \| nontext prompt burden=100% \| repetitive token=unt | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=32.07%. |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | hit token cap (500) \| output/prompt=32.07% \| nontext prompt burden=100% \| repetitive token=unt | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=32.07%. |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | hit token cap (500) \| output/prompt=47.76% \| nontext prompt burden=100% \| degeneration=character_loop: ' 1.' repeated | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=47.76%. |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | hit token cap (500) \| output/prompt=47.76% \| nontext prompt burden=100% \| repetitive token=phrase: "1. these information is..." | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=47.76%. |
| `mlx-community/pixtral-12b-8bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | hit token cap (500) \| output/prompt=47.76% \| nontext prompt burden=100% \| degeneration=character_loop: '. 0' repeated | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=47.76%. |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `clean` | nontext prompt burden=99% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `clean` | nontext prompt burden=99% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `clean` | nontext prompt burden=71% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | hit token cap (500) \| output/prompt=11.74% \| nontext prompt burden=100% \| reasoning leak | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=11.74%. |
| `mlx-community/InternVL3-8B-bf16` | `clean` | nontext prompt burden=100% \| repetitive token=phrase: "rencontre rencontre rencontre ..." | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `clean` | hit token cap (500) \| nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Molmo-7B-D-0924-8bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/X-Reasoner-7B-8bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%. |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `clean` | nontext prompt burden=73% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `cutoff` | hit token cap (500) \| output/prompt=48.59% \| nontext prompt burden=100% \| repetitive token=phrase: "the light fixtures are..." | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=48.59%. |
| `mlx-community/pixtral-12b-bf16` | `clean` | hit token cap (500) \| nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Molmo-7B-D-0924-bf16` | `cutoff` | hit token cap (500) \| output/prompt=41.70% \| nontext prompt burden=100% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=41.70%. |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%. |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% \| degeneration=incomplete_sentence: ends with 's' | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%. |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%. |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% \| degeneration=incomplete_sentence: ends with 'of' | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%. |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | hit token cap (500) \| output/prompt=3.07% \| nontext prompt burden=100% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%. |

### `model-config`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | processor error \| model config processor load processor | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/gemma-3n-E2B-4bit` | `harness` | Output appears truncated to about 2 tokens. \| nontext prompt burden=98% | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | Output appears truncated to about 2 tokens. \| nontext prompt burden=100% | Inspect model repo config, chat template, and EOS settings. |

## Model Verdicts

### `mlx-community/MolmoPoint-8B-fp16`

- **Verdict:** harness | user=avoid
- **Why:** processor error | model config processor load processor
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=model-config | package=model-config | stage=Processor Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=271 | text_est=4 | nontext_est=267 | gen=7 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** harness | user=avoid
- **Why:** Output appears truncated to about 2 tokens. | nontext prompt burden=98%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=264 | text_est=4 | nontext_est=260 | gen=2 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=768 | text_est=4 | nontext_est=764 | gen=16 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=80%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=20 | text_est=4 | nontext_est=16 | gen=168 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=95%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=79 | text_est=4 | nontext_est=75 | gen=137 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** Output is very short relative to prompt size (0.3%), suggesting possible early-stop or prompt-handling issues. | At long prompt length (4101 tokens), output stayed unusually short (12 tokens; ratio 0.3%). | output/prompt=0.29% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=caveat | preserves trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=4101 | text_est=4 | nontext_est=4097 | gen=12 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output appears truncated to about 2 tokens. | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1029 | text_est=4 | nontext_est=1025 | gen=2 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=83%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=24 | text_est=4 | nontext_est=20 | gen=420 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3619 | text_est=4 | nontext_est=3615 | gen=202 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2942 | text_est=4 | nontext_est=2938 | gen=98 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2572 | text_est=4 | nontext_est=2568 | gen=50 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** clean | user=recommended
- **Why:** hit token cap (500) | nontext prompt burden=96%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=95 | text_est=4 | nontext_est=91 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=272 | text_est=4 | nontext_est=268 | gen=204 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2500.00% | nontext prompt burden=80% | repetitive token=Baz
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=20 | text_est=4 | nontext_est=16 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2500.00%.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=32.07% | nontext prompt burden=100% | repetitive token=unt
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1559 | text_est=4 | nontext_est=1555 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=32.07%.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=32.07% | nontext prompt burden=100% | repetitive token=unt
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1559 | text_est=4 | nontext_est=1555 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=32.07%.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=47.76% | nontext prompt burden=100% | degeneration=character_loop: ' 1.' repeated
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1047 | text_est=4 | nontext_est=1043 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=47.76%.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1029 | text_est=4 | nontext_est=1025 | gen=142 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=47.76% | nontext prompt burden=100% | repetitive token=phrase: "1. these information is..."
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1047 | text_est=4 | nontext_est=1043 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=47.76%.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4167 | text_est=4 | nontext_est=4163 | gen=136 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3620 | text_est=4 | nontext_est=3616 | gen=193 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=47.76% | nontext prompt burden=100% | degeneration=character_loop: '. 0' repeated
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1047 | text_est=4 | nontext_est=1043 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=47.76%.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=273 | text_est=4 | nontext_est=269 | gen=119 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=273 | text_est=4 | nontext_est=269 | gen=212 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3620 | text_est=4 | nontext_est=3616 | gen=279 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|end|&gt; appeared in generated text. | Special control token &lt;|endoftext|&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=768 | text_est=4 | nontext_est=764 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 127 occurrences). | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | generic
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=3087 | text_est=4 | nontext_est=3083 | gen=160 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6095 | text_est=4 | nontext_est=6091 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=71%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=14 | text_est=4 | nontext_est=10 | gen=213 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=100% | degeneration=incomplete_sentence: ends with 'a'
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1271 | text_est=4 | nontext_est=1267 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6095 | text_est=4 | nontext_est=6091 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=11.74% | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4259 | text_est=4 | nontext_est=4255 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=11.74%.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100% | repetitive token=phrase: "rencontre rencontre rencontre ..."
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2572 | text_est=4 | nontext_est=2568 | gen=434 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** clean | user=recommended
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3038 | text_est=4 | nontext_est=3034 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1199 | text_est=4 | nontext_est=1195 | gen=90 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16280 | text_est=4 | nontext_est=16276 | gen=218 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.07% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16271 | text_est=4 | nontext_est=16267 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** At long prompt length (16269 tokens), output became repetitive. | hit token cap (500) | output/prompt=3.07% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16269 | text_est=4 | nontext_est=16265 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=73%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=15 | text_est=4 | nontext_est=11 | gen=112 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=48.59% | nontext prompt burden=100% | repetitive token=phrase: "the light fixtures are..."
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1029 | text_est=4 | nontext_est=1025 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=48.59%.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** clean | user=recommended
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4167 | text_est=4 | nontext_est=4163 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6095 | text_est=4 | nontext_est=6091 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=41.70% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1199 | text_est=4 | nontext_est=1195 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=41.70%.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.07% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16271 | text_est=4 | nontext_est=16267 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** Output appears truncated to about 4 tokens. | At long prompt length (16280 tokens), output stayed unusually short (4 tokens; ratio 0.0%). | output/prompt=0.02% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=caveat | preserves trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16280 | text_est=4 | nontext_est=16276 | gen=4 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.07% | nontext prompt burden=100% | degeneration=incomplete_sentence: ends with 's'
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16271 | text_est=4 | nontext_est=16267 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.07% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16271 | text_est=4 | nontext_est=16267 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.07% | nontext prompt burden=100% | degeneration=incomplete_sentence: ends with 'of'
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16271 | text_est=4 | nontext_est=16267 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=16271 | text_est=4 | nontext_est=16267 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.07% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16271 | text_est=4 | nontext_est=16267 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.07%.
