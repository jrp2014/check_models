# Automated Review Digest

_Generated on 2026-04-12 21:48:31 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [check_models.log](check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🏆 A (93/100) | Desc 93 | Keywords 93 | Δ+77 | 185.6 tps
- `mlx-community/pixtral-12b-8bit`: 🏆 A (93/100) | Desc 93 | Keywords 89 | Δ+77 | 39.2 tps

### Watchlist

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-16 | harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | Δ-10 | 30.9 tps | harness, metadata borrowing, missing sections
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) | Desc 57 | Keywords 0 | Δ-5 | 71.4 tps | context ignored, harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (19/100) | Desc 46 | Keywords 42 | Δ+3 | 5.6 tps | context ignored, harness, missing sections
- `Qwen/Qwen3-VL-2B-Instruct`: 🟡 C (59/100) | Desc 76 | Keywords 55 | Δ+42 | 86.8 tps | context ignored, cutoff, harness, long context, missing sections, repetitive

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

- None.

### `caveat`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | improves trusted hints | nontext prompt burden=87% \| missing terms: United, Kingdom |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `clean` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | nontext prompt burden=69% \| missing terms: Town, Centre, Alton, United, Kingdom \| keywords=22 |
| `mlx-community/InternVL3-14B-8bit` | `clean` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | nontext prompt burden=83% \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/pixtral-12b-8bit` | `clean` | improves trusted hints | nontext prompt burden=88% \| missing terms: Alton, United, Kingdom |
| `mlx-community/X-Reasoner-7B-8bit` | `context_budget` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | At long prompt length (16680 tokens), output may stop following prompt/image context. \| output/prompt=0.83% \| nontext prompt burden=98% \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/Qwen3.5-27B-mxfp8` | `context_budget` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing terms: United, Kingdom |

### `avoid`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | not evaluated | processor error \| model config processor load processor |
| `mlx-community/LFM2-VL-1.6B-8bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | missing sections: title, description, keywords \| nonvisual metadata reused \| reasoning leak |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | nonvisual metadata reused |
| `mlx-community/nanoLLaVA-1.5-4bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | keywords=32 \| nonvisual metadata reused \| reasoning leak |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `model_shortcoming` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/FastVLM-0.5B-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom \| nonvisual metadata reused | hit token cap (500) \| output/prompt=108.23% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `harness` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | Output appears truncated to about 7 tokens. \| nontext prompt burden=85% \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/gemma-3n-E4B-it-bf16` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | missing sections: title, description, keywords \| missing terms: Town, Centre, United, Kingdom \| nonvisual metadata reused |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | missing terms: United, Kingdom \| nonvisual metadata reused |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | nontext prompt burden=87% \| missing terms: Alton, United, Kingdom \| nonvisual metadata reused |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=88.18% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | nontext prompt burden=87% \| missing terms: Alton, United, Kingdom \| nonvisual metadata reused |
| `mlx-community/gemma-3n-E2B-4bit` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=69.54% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=30.01% \| nontext prompt burden=77% \| missing sections: title, description, keywords |
| `qnguyen3/nanoLLaVA` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=109.17% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=30.01% \| nontext prompt burden=77% \| missing sections: title, description, keywords |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=35.41% \| nontext prompt burden=73% \| missing sections: title, description, keywords |
| `mlx-community/gemma-4-31b-it-4bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | missing terms: Town, Centre, United, Kingdom \| nonvisual metadata reused |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=35.41% \| nontext prompt burden=73% \| missing sections: title, description, keywords |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | missing terms: United, Kingdom \| nonvisual metadata reused |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | preserves trusted hints \| nonvisual metadata reused | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 76 occurrences). \| nontext prompt burden=84% \| missing sections: description, keywords \| nonvisual metadata reused |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=35.41% \| nontext prompt burden=73% \| missing sections: title, description, keywords |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | missing sections: title, description, keywords \| missing terms: United, Kingdom \| nonvisual metadata reused |
| `microsoft/Phi-3.5-vision-instruct` | `harness` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom \| nonvisual metadata reused | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=69% |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=7.76% \| nontext prompt burden=94% \| missing sections: title, description, keywords |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=10.97% \| nontext prompt burden=92% \| missing sections: title, description, keywords |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=15.31% \| nontext prompt burden=88% \| missing sections: title, description, keywords |
| `mlx-community/GLM-4.6V-Flash-6bit` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=7.76% \| nontext prompt burden=94% \| missing sections: title, description, keywords |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `harness` | improves trusted hints \| nonvisual metadata reused | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=78% \| missing terms: Centre |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=18.25% \| nontext prompt burden=86% \| missing sections: title, description, keywords |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | At long prompt length (16669 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% |
| `mlx-community/Molmo-7B-D-0924-8bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | nontext prompt burden=76% \| missing terms: Town, Centre \| nonvisual metadata reused \| reasoning leak |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: title, description, keywords |
| `mlx-community/pixtral-12b-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=15.75% \| nontext prompt burden=88% \| missing sections: title, description, keywords |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=33.69% \| nontext prompt burden=74% \| missing sections: title, description, keywords |
| `mlx-community/InternVL3-8B-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=22.31% \| nontext prompt burden=83% \| missing sections: title, description, keywords |
| `mlx-community/GLM-4.6V-nvfp4` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=7.76% \| nontext prompt burden=94% \| missing sections: title, description, keywords |
| `mlx-community/Molmo-7B-D-0924-bf16` | `model_shortcoming` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | nontext prompt burden=76% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: title, description, keywords |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `harness` | ignores trusted hints | Model returned zero output tokens. |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: title, description, keywords |
| `mlx-community/gemma-4-31b-bf16` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=69.35% \| missing sections: title, description, keywords \| nonvisual metadata reused |
| `mlx-community/Qwen3.5-27B-4bit` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: keywords |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=116.28% \| missing sections: title, description, keywords \| missing terms: Centre, United, Kingdom |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `mlx`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | At long prompt length (16669 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% | Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete. |
| `mlx-community/X-Reasoner-7B-8bit` | `context_budget` | At long prompt length (16680 tokens), output may stop following prompt/image context. \| output/prompt=0.83% \| nontext prompt burden=98% \| missing terms: Town, Centre, Alton, United, Kingdom | Treat this as a prompt-budget issue first; nontext prompt burden is 98% and the output stays weak under that load. |

### `mlx-vlm`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 76 occurrences). \| nontext prompt burden=84% \| missing sections: description, keywords \| nonvisual metadata reused | Inspect decode cleanup; tokenizer markers are leaking into user-facing text. |
| `microsoft/Phi-3.5-vision-instruct` | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=69% | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=78% \| missing terms: Centre | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/LFM2-VL-1.6B-8bit` | `model_shortcoming` | missing sections: title, description, keywords \| nonvisual metadata reused \| reasoning leak | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `model_shortcoming` | nonvisual metadata reused | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/nanoLLaVA-1.5-4bit` | `model_shortcoming` | keywords=32 \| nonvisual metadata reused \| reasoning leak | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | nontext prompt burden=87% \| missing terms: United, Kingdom | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `model_shortcoming` | nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/FastVLM-0.5B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=108.23% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-3n-E4B-it-bf16` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Town, Centre, United, Kingdom \| nonvisual metadata reused | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `clean` | nontext prompt burden=69% \| missing terms: Town, Centre, Alton, United, Kingdom \| keywords=22 | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/InternVL3-14B-8bit` | `clean` | nontext prompt burden=83% \| missing terms: Town, Centre, Alton, United, Kingdom | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `model_shortcoming` | missing terms: United, Kingdom \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `model_shortcoming` | nontext prompt burden=87% \| missing terms: Alton, United, Kingdom \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `cutoff` | hit token cap (500) \| output/prompt=88.18% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `model_shortcoming` | nontext prompt burden=87% \| missing terms: Alton, United, Kingdom \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-3n-E2B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=69.54% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | hit token cap (500) \| output/prompt=30.01% \| nontext prompt burden=77% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `qnguyen3/nanoLLaVA` | `cutoff` | hit token cap (500) \| output/prompt=109.17% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | hit token cap (500) \| output/prompt=30.01% \| nontext prompt burden=77% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/pixtral-12b-8bit` | `clean` | nontext prompt burden=88% \| missing terms: Alton, United, Kingdom | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | hit token cap (500) \| output/prompt=35.41% \| nontext prompt burden=73% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-4-31b-it-4bit` | `model_shortcoming` | missing terms: Town, Centre, United, Kingdom \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | hit token cap (500) \| output/prompt=35.41% \| nontext prompt burden=73% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `model_shortcoming` | missing terms: United, Kingdom \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | hit token cap (500) \| output/prompt=35.41% \| nontext prompt burden=73% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: United, Kingdom \| nonvisual metadata reused | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `cutoff` | hit token cap (500) \| output/prompt=7.76% \| nontext prompt burden=94% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `cutoff` | hit token cap (500) \| output/prompt=10.97% \| nontext prompt burden=92% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | hit token cap (500) \| output/prompt=15.31% \| nontext prompt burden=88% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/GLM-4.6V-Flash-6bit` | `cutoff` | hit token cap (500) \| output/prompt=7.76% \| nontext prompt burden=94% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `cutoff` | hit token cap (500) \| output/prompt=18.25% \| nontext prompt burden=86% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Molmo-7B-D-0924-8bit` | `model_shortcoming` | nontext prompt burden=76% \| missing terms: Town, Centre \| nonvisual metadata reused \| reasoning leak | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/pixtral-12b-bf16` | `cutoff` | hit token cap (500) \| output/prompt=15.75% \| nontext prompt burden=88% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `cutoff` | hit token cap (500) \| output/prompt=33.69% \| nontext prompt burden=74% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/InternVL3-8B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=22.31% \| nontext prompt burden=83% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/GLM-4.6V-nvfp4` | `cutoff` | hit token cap (500) \| output/prompt=7.76% \| nontext prompt burden=94% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Molmo-7B-D-0924-bf16` | `model_shortcoming` | nontext prompt burden=76% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-4-31b-bf16` | `cutoff` | hit token cap (500) \| output/prompt=69.35% \| missing sections: title, description, keywords \| nonvisual metadata reused | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-27B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing sections: keywords | Raise the token cap or trim prompt burden first; generation hit the limit while keywords remained incomplete. |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `cutoff` | hit token cap (500) \| output/prompt=116.28% \| missing sections: title, description, keywords \| missing terms: Centre, United, Kingdom | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-27B-mxfp8` | `context_budget` | hit token cap (500) \| output/prompt=3.00% \| nontext prompt burden=98% \| missing terms: United, Kingdom | Treat this as a prompt-budget issue first; nontext prompt burden is 98% and the output stays weak under that load. |

### `model-config`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | processor error \| model config processor load processor | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `harness` | Output appears truncated to about 7 tokens. \| nontext prompt burden=85% \| missing terms: Town, Centre, Alton, United, Kingdom | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Check chat-template and EOS defaults first; the output shape is not matching the requested contract. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `harness` | Model returned zero output tokens. | Inspect model repo config, chat template, and EOS settings. |

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

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | nonvisual metadata reused | reasoning leak
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=711 | text_est=387 | nontext_est=324 | gen=85 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** description sentences=3
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=521 | text_est=387 | nontext_est=134 | gen=127 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** keywords=32 | nonvisual metadata reused | reasoning leak
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=22 | description sentences=3 | keywords=32
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=458 | text_est=387 | nontext_est=71 | gen=243 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=87% | missing terms: United, Kingdom
- **Trusted hints:** improves trusted hints
- **Contract:** description sentences=3
- **Utility:** user=caveat | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2937 | text_est=387 | nontext_est=2550 | gen=143 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1484 | text_est=387 | nontext_est=1097 | gen=25 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=108.23% | missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=462 | text_est=387 | nontext_est=75 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** harness | user=avoid
- **Why:** Output appears truncated to about 7 tokens. | nontext prompt burden=85% | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=2554 | text_est=387 | nontext_est=2167 | gen=7 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Town, Centre, United, Kingdom | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=727 | text_est=387 | nontext_est=340 | gen=106 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=69% | missing terms: Town, Centre, Alton, United, Kingdom | keywords=22
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** keywords=22
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1265 | text_est=387 | nontext_est=878 | gen=130 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=83% | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** title words=2
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2241 | text_est=387 | nontext_est=1854 | gen=46 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1484 | text_est=387 | nontext_est=1097 | gen=12 | max=500 | stop=completed
- **Next action:** Check chat-template and EOS defaults first; the output shape is not matching the requested contract.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: United, Kingdom | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=728 | text_est=387 | nontext_est=341 | gen=85 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=87% | missing terms: Alton, United, Kingdom | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2938 | text_est=387 | nontext_est=2551 | gen=117 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=88.18% | missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=567 | text_est=387 | nontext_est=180 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=87% | missing terms: Alton, United, Kingdom | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** description sentences=3
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2938 | text_est=387 | nontext_est=2551 | gen=143 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=69.54% | missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=719 | text_est=387 | nontext_est=332 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=30.01% | nontext prompt burden=77% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1666 | text_est=387 | nontext_est=1279 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=109.17% | missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=458 | text_est=387 | nontext_est=71 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=30.01% | nontext prompt burden=77% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1666 | text_est=387 | nontext_est=1279 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=88% | missing terms: Alton, United, Kingdom
- **Trusted hints:** improves trusted hints
- **Contract:** description sentences=3
- **Utility:** user=caveat | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3175 | text_est=387 | nontext_est=2788 | gen=112 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=35.41% | nontext prompt burden=73% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1412 | text_est=387 | nontext_est=1025 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/gemma-4-31b-it-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: Town, Centre, United, Kingdom | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=734 | text_est=387 | nontext_est=347 | gen=101 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=35.41% | nontext prompt burden=73% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1412 | text_est=387 | nontext_est=1025 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: United, Kingdom | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=728 | text_est=387 | nontext_est=341 | gen=98 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 76 occurrences). | nontext prompt burden=84% | missing sections: description, keywords | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=79
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2439 | text_est=387 | nontext_est=2052 | gen=108 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=35.41% | nontext prompt burden=73% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1412 | text_est=387 | nontext_est=1025 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: United, Kingdom | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=429 | text_est=387 | nontext_est=42 | gen=126 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|end|&gt; appeared in generated text. | Special control token &lt;|endoftext|&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=69%
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom | nonvisual metadata reused
- **Contract:** keywords=52
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1265 | text_est=387 | nontext_est=878 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.76% | nontext prompt burden=94% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6444 | text_est=387 | nontext_est=6057 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=10.97% | nontext prompt burden=92% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4556 | text_est=387 | nontext_est=4169 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=15.31% | nontext prompt burden=88% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3266 | text_est=387 | nontext_est=2879 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.76% | nontext prompt burden=94% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6444 | text_est=387 | nontext_est=6057 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=78% | missing terms: Centre
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** keywords=9
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1754 | text_est=387 | nontext_est=1367 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=18.25% | nontext prompt burden=86% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2740 | text_est=387 | nontext_est=2353 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** At long prompt length (16669 tokens), output became repetitive. | hit token cap (500) | output/prompt=3.00% | nontext prompt burden=98%
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: description, keywords | title words=302
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16669 | text_est=387 | nontext_est=16282 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** context_budget | user=caveat
- **Why:** At long prompt length (16680 tokens), output may stop following prompt/image context. | output/prompt=0.83% | nontext prompt burden=98% | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** title words=4 | description sentences=3 | keywords=21
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16680 | text_est=387 | nontext_est=16293 | gen=138 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 98% and the output stays weak under that load.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=76% | missing terms: Town, Centre | nonvisual metadata reused | reasoning leak
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** description sentences=4
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1637 | text_est=387 | nontext_est=1250 | gen=126 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.00% | nontext prompt burden=98% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16671 | text_est=387 | nontext_est=16284 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=15.75% | nontext prompt burden=88% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3175 | text_est=387 | nontext_est=2788 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=33.69% | nontext prompt burden=74% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1484 | text_est=387 | nontext_est=1097 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=22.31% | nontext prompt burden=83% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2241 | text_est=387 | nontext_est=1854 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.76% | nontext prompt burden=94% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6444 | text_est=387 | nontext_est=6057 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=76% | missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1637 | text_est=387 | nontext_est=1250 | gen=485 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.00% | nontext prompt burden=98% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16694 | text_est=387 | nontext_est=16307 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.00% | nontext prompt burden=98% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16694 | text_est=387 | nontext_est=16307 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** harness | user=avoid
- **Why:** Model returned zero output tokens.
- **Trusted hints:** ignores trusted hints
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=0 | text_est=387 | nontext_est=0 | gen=0 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.00% | nontext prompt burden=98% | missing sections: description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16694 | text_est=387 | nontext_est=16307 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.00% | nontext prompt burden=98% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16694 | text_est=387 | nontext_est=16307 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/gemma-4-31b-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=69.35% | missing sections: title, description, keywords | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=721 | text_est=387 | nontext_est=334 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=3.00% | nontext prompt burden=98% | missing sections: keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords | title words=50
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16694 | text_est=387 | nontext_est=16307 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while keywords remained incomplete.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=116.28% | missing sections: title, description, keywords | missing terms: Centre, United, Kingdom
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=430 | text_est=387 | nontext_est=43 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** context_budget | user=caveat
- **Why:** hit token cap (500) | output/prompt=3.00% | nontext prompt burden=98% | missing terms: United, Kingdom
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=41 | description sentences=4 | keywords=34 | keyword duplication=0.41
- **Utility:** user=caveat | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16694 | text_est=387 | nontext_est=16307 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 98% and the output stays weak under that load.
