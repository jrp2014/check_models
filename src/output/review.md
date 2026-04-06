# Automated Review Digest

_Generated on 2026-04-06 17:15:49 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [check_models.log](check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/Molmo-7B-D-0924-8bit`: 🏆 A (90/100) | 52.4 tps
- `mlx-community/Qwen3.5-9B-MLX-4bit`: 🏆 A (85/100) | 89.6 tps
- `mlx-community/nanoLLaVA-1.5-4bit`: 🏆 A (80/100) | 344.5 tps
- `mlx-community/LFM2.5-VL-1.6B-bf16`: 🏆 A (80/100) | 187.9 tps
- `mlx-community/gemma-3n-E4B-it-bf16`: 🏆 A (80/100) | 47.8 tps

### Watchlist

- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (0/100) | 333.0 tps | harness
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (0/100) | 76.1 tps | harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (0/100) | 10.6 tps | harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | 31.5 tps | harness
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) | 223.8 tps | harness, long context

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `clean` | preserves trusted hints | nontext prompt burden=99% |
| `mlx-community/nanoLLaVA-1.5-4bit` | `clean` | preserves trusted hints | nontext prompt burden=80% |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `clean` | preserves trusted hints | nontext prompt burden=95% |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/InternVL3-14B-8bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `clean` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=96% |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/pixtral-12b-8bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/gemma-3n-E4B-it-bf16` | `clean` | preserves trusted hints | nontext prompt burden=99% |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `clean` | preserves trusted hints | nontext prompt burden=99% |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `clean` | preserves trusted hints | nontext prompt burden=71% |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `clean` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100% |
| `mlx-community/InternVL3-8B-bf16` | `clean` | preserves trusted hints | nontext prompt burden=100% \| repetitive token=phrase: "strugg strugg strugg strugg..." |
| `mlx-community/Molmo-7B-D-0924-8bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/X-Reasoner-7B-8bit` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/pixtral-12b-bf16` | `clean` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100% |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `clean` | preserves trusted hints | nontext prompt burden=73% |
| `mlx-community/Molmo-7B-D-0924-bf16` | `clean` | preserves trusted hints | nontext prompt burden=100% |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `clean` | preserves trusted hints | hit token cap (500) \| nontext prompt burden=100% |

### `caveat`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | preserves trusted hints | Output appears truncated to about 6 tokens. \| At long prompt length (16248 tokens), output stayed unusually short (6 tokens; ratio 0.0%). \| output/prompt=0.04% \| nontext prompt burden=100% |

### `avoid`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | not evaluated | processor error \| model config processor load processor |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `harness` | not evaluated | model error \| huggingface hub model load model \| hub connectivity |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `harness` | not evaluated | model error \| huggingface hub model load model \| hub connectivity |
| `mlx-community/LFM2-VL-1.6B-8bit` | `harness` | preserves trusted hints | Output appears truncated to about 4 tokens. \| nontext prompt burden=99% |
| `mlx-community/gemma-3n-E2B-4bit` | `harness` | preserves trusted hints | Output appears truncated to about 2 tokens. \| nontext prompt burden=98% |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | preserves trusted hints | Output appears truncated to about 2 tokens. \| nontext prompt burden=100% |
| `mlx-community/FastVLM-0.5B-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=2083.33% \| nontext prompt burden=83% |
| `qnguyen3/nanoLLaVA` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=2500.00% \| nontext prompt burden=80% \| repetitive token=Baz |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=48.22% \| nontext prompt burden=100% \| repetitive token=答案内容1. |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=41.84% \| nontext prompt burden=100% \| repetitive token=phrase: "treasured treasured treasured ..." |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=41.84% \| nontext prompt burden=100% \| repetitive token=phrase: "treasured treasured treasured ..." |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=48.22% \| nontext prompt burden=100% \| repetitive token=phrase: "will be used to..." |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | preserves trusted hints | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 103 occurrences). \| nontext prompt burden=100% |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=48.22% \| nontext prompt burden=100% \| degeneration=character_loop: '. 0' repeated |
| `microsoft/Phi-3.5-vision-instruct` | `harness` | preserves trusted hints | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=99% |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `harness` | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| nontext prompt burden=100% \| reasoning leak |
| `mlx-community/GLM-4.6V-Flash-6bit` | `harness` | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| nontext prompt burden=100% \| reasoning leak |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=38.67% \| nontext prompt burden=100% |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=17.15% \| nontext prompt burden=100% \| reasoning leak |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=3.08% \| nontext prompt burden=100% \| degeneration=character_loop: '0' repeated |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | preserves trusted hints | At long prompt length (16239 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=3.08% \| nontext prompt burden=100% |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=48.59% \| nontext prompt burden=100% \| repetitive token=phrase: "black and white and..." |
| `mlx-community/GLM-4.6V-nvfp4` | `harness` | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=183.15% \| nontext prompt burden=99% \| degeneration=character_loop: '00' repeated |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=3.08% \| nontext prompt burden=100% |
| `mlx-community/Qwen3.5-27B-4bit` | `harness` | preserves trusted hints | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=3.08% \| nontext prompt burden=100% |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `huggingface-hub`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `harness` | model error \| huggingface hub model load model \| hub connectivity | Check whether Hugging Face was reachable; this may be a transient Hub/network outage or disconnect rather than a model defect. |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `harness` | model error \| huggingface hub model load model \| hub connectivity | Check whether Hugging Face was reachable; this may be a transient Hub/network outage or disconnect rather than a model defect. |

### `mlx`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | At long prompt length (16239 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=3.08% \| nontext prompt burden=100% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.08%. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | Output appears truncated to about 6 tokens. \| At long prompt length (16248 tokens), output stayed unusually short (6 tokens; ratio 0.0%). \| output/prompt=0.04% \| nontext prompt burden=100% | Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load. |

### `mlx-vlm`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 103 occurrences). \| nontext prompt burden=100% | Inspect decode cleanup; tokenizer markers are leaking into user-facing text. |
| `microsoft/Phi-3.5-vision-instruct` | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=99% | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| nontext prompt burden=100% \| reasoning leak | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-Flash-6bit` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| nontext prompt burden=100% \| reasoning leak | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-nvfp4` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Qwen3.5-27B-4bit` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=100% | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `clean` | nontext prompt burden=99% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/nanoLLaVA-1.5-4bit` | `clean` | nontext prompt burden=80% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `clean` | nontext prompt burden=95% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/FastVLM-0.5B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=2083.33% \| nontext prompt burden=83% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2083.33%. |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/InternVL3-14B-8bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `clean` | hit token cap (500) \| nontext prompt burden=96% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/pixtral-12b-8bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `qnguyen3/nanoLLaVA` | `cutoff` | hit token cap (500) \| output/prompt=2500.00% \| nontext prompt burden=80% \| repetitive token=Baz | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2500.00%. |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | hit token cap (500) \| output/prompt=48.22% \| nontext prompt burden=100% \| repetitive token=答案内容1. | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=48.22%. |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | hit token cap (500) \| output/prompt=41.84% \| nontext prompt burden=100% \| repetitive token=phrase: "treasured treasured treasured ..." | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=41.84%. |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | hit token cap (500) \| output/prompt=41.84% \| nontext prompt burden=100% \| repetitive token=phrase: "treasured treasured treasured ..." | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=41.84%. |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | hit token cap (500) \| output/prompt=48.22% \| nontext prompt burden=100% \| repetitive token=phrase: "will be used to..." | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=48.22%. |
| `mlx-community/gemma-3n-E4B-it-bf16` | `clean` | nontext prompt burden=99% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `clean` | nontext prompt burden=99% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `clean` | nontext prompt burden=71% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | hit token cap (500) \| output/prompt=48.22% \| nontext prompt burden=100% \| degeneration=character_loop: '. 0' repeated | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=48.22%. |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff` | hit token cap (500) \| output/prompt=38.67% \| nontext prompt burden=100% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=38.67%. |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | hit token cap (500) \| output/prompt=17.15% \| nontext prompt burden=100% \| reasoning leak | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=17.15%. |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `clean` | hit token cap (500) \| nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/InternVL3-8B-bf16` | `clean` | nontext prompt burden=100% \| repetitive token=phrase: "strugg strugg strugg strugg..." | Treat as a model-quality limitation for this prompt and image. |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | hit token cap (500) \| output/prompt=3.08% \| nontext prompt burden=100% \| degeneration=character_loop: '0' repeated | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.08%. |
| `mlx-community/Molmo-7B-D-0924-8bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `cutoff` | hit token cap (500) \| output/prompt=48.59% \| nontext prompt burden=100% \| repetitive token=phrase: "black and white and..." | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=48.59%. |
| `mlx-community/X-Reasoner-7B-8bit` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `cutoff` | hit token cap (500) \| output/prompt=183.15% \| nontext prompt burden=99% \| degeneration=character_loop: '00' repeated | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=183.15%. |
| `mlx-community/pixtral-12b-bf16` | `clean` | hit token cap (500) \| nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `clean` | nontext prompt burden=73% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Molmo-7B-D-0924-bf16` | `clean` | nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `clean` | hit token cap (500) \| nontext prompt burden=100% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=3.08% \| nontext prompt burden=100% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.08%. |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | hit token cap (500) \| output/prompt=3.08% \| nontext prompt burden=100% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.08%. |

### `model-config`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | processor error \| model config processor load processor | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/LFM2-VL-1.6B-8bit` | `harness` | Output appears truncated to about 4 tokens. \| nontext prompt burden=99% | Inspect model repo config, chat template, and EOS settings. |
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

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** harness | user=avoid
- **Why:** model error | huggingface hub model load model | hub connectivity
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=huggingface-hub | package=huggingface-hub | stage=Model Error | code=HUGGINGFACE_HUB_MODEL_LOAD_MODEL
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Check whether Hugging Face was reachable; this may be a transient Hub/network outage or disconnect rather than a model defect.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** harness | user=avoid
- **Why:** model error | huggingface hub model load model | hub connectivity
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=huggingface-hub | package=huggingface-hub | stage=Model Error | code=HUGGINGFACE_HUB_MODEL_LOAD_MODEL
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Check whether Hugging Face was reachable; this may be a transient Hub/network outage or disconnect rather than a model defect.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** harness | user=avoid
- **Why:** Output appears truncated to about 4 tokens. | nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=275 | text_est=4 | nontext_est=271 | gen=4 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

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
- **Token accounting:** prompt=20 | text_est=4 | nontext_est=16 | gen=147 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output appears truncated to about 2 tokens. | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1029 | text_est=4 | nontext_est=1025 | gen=2 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=95%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=85 | text_est=4 | nontext_est=81 | gen=287 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4101 | text_est=4 | nontext_est=4097 | gen=22 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2083.33% | nontext prompt burden=83%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=24 | text_est=4 | nontext_est=20 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2083.33%.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2629 | text_est=4 | nontext_est=2625 | gen=342 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1804 | text_est=4 | nontext_est=1800 | gen=69 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2174 | text_est=4 | nontext_est=2170 | gen=54 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2630 | text_est=4 | nontext_est=2626 | gen=133 | max=500 | stop=completed
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

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2630 | text_est=4 | nontext_est=2626 | gen=166 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2823 | text_est=4 | nontext_est=2819 | gen=109 | max=500 | stop=completed
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

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=48.22% | nontext prompt burden=100% | repetitive token=答案内容1.
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1037 | text_est=4 | nontext_est=1033 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=48.22%.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=41.84% | nontext prompt burden=100% | repetitive token=phrase: "treasured treasured treasured ..."
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1195 | text_est=4 | nontext_est=1191 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=41.84%.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=41.84% | nontext prompt burden=100% | repetitive token=phrase: "treasured treasured treasured ..."
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1195 | text_est=4 | nontext_est=1191 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=41.84%.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1029 | text_est=4 | nontext_est=1025 | gen=127 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=48.22% | nontext prompt burden=100% | repetitive token=phrase: "will be used to..."
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1037 | text_est=4 | nontext_est=1033 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=48.22%.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=272 | text_est=4 | nontext_est=268 | gen=240 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=273 | text_est=4 | nontext_est=269 | gen=174 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 103 occurrences). | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2097 | text_est=4 | nontext_est=2093 | gen=119 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=71%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=14 | text_est=4 | nontext_est=10 | gen=121 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=48.22% | nontext prompt burden=100% | degeneration=character_loop: '. 0' repeated
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1037 | text_est=4 | nontext_est=1033 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=48.22%.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|end|&gt; appeared in generated text. | Special control token &lt;|endoftext|&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=99%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=768 | text_est=4 | nontext_est=764 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6155 | text_est=4 | nontext_est=6151 | gen=319 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6155 | text_est=4 | nontext_est=6151 | gen=303 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=38.67% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1293 | text_est=4 | nontext_est=1289 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=38.67%.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=17.15% | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2915 | text_est=4 | nontext_est=2911 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=17.15%.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** clean | user=recommended
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2326 | text_est=4 | nontext_est=2322 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100% | repetitive token=phrase: "strugg strugg strugg strugg..."
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1804 | text_est=4 | nontext_est=1800 | gen=329 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.08% | nontext prompt burden=100% | degeneration=character_loop: '0' repeated
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16237 | text_est=4 | nontext_est=16233 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.08%.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** At long prompt length (16239 tokens), output became repetitive. | hit token cap (500) | output/prompt=3.08% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16239 | text_est=4 | nontext_est=16235 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.08%.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1207 | text_est=4 | nontext_est=1203 | gen=290 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=48.59% | nontext prompt burden=100% | repetitive token=phrase: "black and white and..."
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1029 | text_est=4 | nontext_est=1025 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=48.59%.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16248 | text_est=4 | nontext_est=16244 | gen=476 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=100% | reasoning leak
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6155 | text_est=4 | nontext_est=6151 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=183.15% | nontext prompt burden=99% | degeneration=character_loop: '00' repeated
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=273 | text_est=4 | nontext_est=269 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=183.15%.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** clean | user=recommended
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2823 | text_est=4 | nontext_est=2819 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=73%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=15 | text_est=4 | nontext_est=11 | gen=193 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1207 | text_est=4 | nontext_est=1203 | gen=358 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** Output appears truncated to about 6 tokens. | At long prompt length (16248 tokens), output stayed unusually short (6 tokens; ratio 0.0%). | output/prompt=0.04% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=caveat | preserves trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16248 | text_est=4 | nontext_est=16244 | gen=6 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 100% and the output stays weak under that load.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** clean | user=recommended
- **Why:** hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16239 | text_est=4 | nontext_est=16235 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.08% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16239 | text_est=4 | nontext_est=16235 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.08%.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=16239 | text_est=4 | nontext_est=16235 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.08% | nontext prompt burden=100%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16239 | text_est=4 | nontext_est=16235 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=3.08%.
