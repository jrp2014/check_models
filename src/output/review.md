# Automated Review Digest

_Generated on 2026-04-17 13:13:03 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [check_models.log](check_models.log)

## 🧭 Review Priorities

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | Δ-10 | 31.5 tps | context ignored, harness, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (19/100) | Desc 46 | Keywords 42 | Δ+3 | 5.7 tps | context ignored, harness, missing sections
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) | Desc 60 | Keywords 0 | Δ-16 | 131.1 tps | context ignored, cutoff, missing sections
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) | Desc 40 | Keywords 0 | Δ-16 | 127.6 tps | context ignored, cutoff, missing sections
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) | Desc 60 | Keywords 0 | Δ-16 | 77.9 tps | context ignored, cutoff, degeneration, missing sections

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

- None.

### `caveat`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | nontext prompt burden=87% \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/InternVL3-14B-8bit` | `clean` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | nontext prompt burden=83% \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | nontext prompt burden=87% \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | nontext prompt burden=87% \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/pixtral-12b-8bit` | `clean` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | nontext prompt burden=88% \| missing terms: Town, Centre, Alton, United, Kingdom \| keywords=19 |

### `avoid`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `Qwen/Qwen3-VL-2B-Instruct` | `harness` | improves trusted hints \| nonvisual metadata reused | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=36 \| context echo=100% \| nonvisual metadata reused |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | not evaluated | processor error \| model config processor load processor |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `harness` | improves trusted hints \| nonvisual metadata reused | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=36 \| context echo=100% \| nonvisual metadata reused |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `harness` | improves trusted hints \| nonvisual metadata reused | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=37 \| context echo=100% \| nonvisual metadata reused |
| `mlx-community/X-Reasoner-7B-8bit` | `harness` | improves trusted hints \| nonvisual metadata reused | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=36 \| context echo=100% \| nonvisual metadata reused |
| `mlx-community/nanoLLaVA-1.5-4bit` | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused | missing sections: keywords \| nonvisual metadata reused |
| `mlx-community/LFM2-VL-1.6B-8bit` | `model_shortcoming` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/FastVLM-0.5B-bf16` | `model_shortcoming` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `model_shortcoming` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom \| nonvisual metadata reused | nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom \| nonvisual metadata reused |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=95.97% \| missing terms: Town, Centre, Alton, United, Kingdom \| keyword duplication=85% |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | missing terms: United, Kingdom \| nonvisual metadata reused |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | nontext prompt burden=85% \| missing sections: title, description, keywords \| nonvisual metadata reused |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=88.18% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/gemma-4-31b-it-4bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | missing terms: Centre, United, Kingdom \| nonvisual metadata reused |
| `qnguyen3/nanoLLaVA` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=109.17% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=29.99% \| nontext prompt burden=77% \| missing sections: title, description, keywords |
| `mlx-community/gemma-3n-E2B-4bit` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom \| nonvisual metadata reused | hit token cap (500) \| output/prompt=69.54% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=34.46% \| nontext prompt burden=73% \| missing sections: title, description, keywords |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 53 occurrences). \| nontext prompt burden=85% \| missing sections: description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=29.99% \| nontext prompt burden=77% \| missing sections: title, description, keywords |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `model_shortcoming` | improves trusted hints | missing sections: title, description, keywords \| missing terms: United, Kingdom |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | missing terms: United, Kingdom \| nonvisual metadata reused |
| `mlx-community/gemma-3n-E4B-it-bf16` | `model_shortcoming` | improves trusted hints | missing sections: title, description, keywords \| missing terms: Centre, Alton, United, Kingdom |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=34.46% \| nontext prompt burden=73% \| missing sections: title, description, keywords |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=34.46% \| nontext prompt burden=73% \| missing sections: title, description, keywords |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=39.53% \| nontext prompt burden=69% \| missing terms: Town, Centre, Alton, United, Kingdom |
| `microsoft/Phi-3.5-vision-instruct` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=39.53% \| nontext prompt burden=69% \| missing terms: Town, Centre, Alton, United, Kingdom |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=10.97% \| nontext prompt burden=92% \| missing sections: title, description, keywords |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=14.92% \| nontext prompt burden=88% \| missing sections: title, keywords |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `cutoff` | improves trusted hints | hit token cap (500) \| output/prompt=7.61% \| nontext prompt burden=94% \| missing sections: title, description, keywords |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=18.24% \| nontext prompt burden=86% \| missing sections: title, description, keywords |
| `mlx-community/InternVL3-8B-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=22.31% \| nontext prompt burden=83% \| missing sections: title, description, keywords |
| `mlx-community/GLM-4.6V-Flash-6bit` | `cutoff` | improves trusted hints | hit token cap (500) \| output/prompt=7.61% \| nontext prompt burden=94% \| missing sections: title, description |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=28.26% \| nontext prompt burden=78% \| missing sections: title, description, keywords |
| `mlx-community/Molmo-7B-D-0924-8bit` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=30.40% \| nontext prompt burden=76% \| missing sections: title, description, keywords |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom \| nonvisual metadata reused | hit token cap (500) \| output/prompt=33.69% \| nontext prompt burden=74% \| missing sections: title, description, keywords |
| `mlx-community/GLM-4.6V-nvfp4` | `cutoff` | improves trusted hints | hit token cap (500) \| output/prompt=7.61% \| nontext prompt burden=94% \| missing sections: title |
| `mlx-community/pixtral-12b-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=15.33% \| nontext prompt burden=88% \| missing sections: title, description, keywords |
| `mlx-community/Molmo-7B-D-0924-bf16` | `cutoff` | ignores trusted hints \| missing terms: Town, Centre, Alton, United, Kingdom | hit token cap (500) \| output/prompt=30.40% \| nontext prompt burden=76% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: title, description, keywords |
| `mlx-community/gemma-4-31b-bf16` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=69.35% \| missing sections: title, description, keywords \| nonvisual metadata reused |
| `mlx-community/Qwen3.5-27B-4bit` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: description, keywords |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=116.28% \| missing sections: title, description, keywords \| missing terms: Centre, United, Kingdom |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | improves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: description, keywords |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `mlx-vlm`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `Qwen/Qwen3-VL-2B-Instruct` | `harness` | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=36 \| context echo=100% \| nonvisual metadata reused | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `harness` | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=36 \| context echo=100% \| nonvisual metadata reused | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `harness` | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=37 \| context echo=100% \| nonvisual metadata reused | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/X-Reasoner-7B-8bit` | `harness` | Special control token &lt;\|im_end\|&gt; appeared in generated text. \| keywords=36 \| context echo=100% \| nonvisual metadata reused | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 53 occurrences). \| nontext prompt burden=85% \| missing sections: description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Inspect decode cleanup; tokenizer markers are leaking into user-facing text. |

### `model`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/nanoLLaVA-1.5-4bit` | `model_shortcoming` | missing sections: keywords \| nonvisual metadata reused | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/LFM2-VL-1.6B-8bit` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/FastVLM-0.5B-bf16` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | nontext prompt burden=87% \| missing terms: Town, Centre, Alton, United, Kingdom | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `model_shortcoming` | nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom \| nonvisual metadata reused | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=95.97% \| missing terms: Town, Centre, Alton, United, Kingdom \| keyword duplication=85% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=95.97%. |
| `mlx-community/InternVL3-14B-8bit` | `clean` | nontext prompt burden=83% \| missing terms: Town, Centre, Alton, United, Kingdom | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `model_shortcoming` | missing terms: United, Kingdom \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `model_shortcoming` | nontext prompt burden=85% \| missing sections: title, description, keywords \| nonvisual metadata reused | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | nontext prompt burden=87% \| missing terms: Town, Centre, Alton, United, Kingdom | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `cutoff` | hit token cap (500) \| output/prompt=88.18% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | nontext prompt burden=87% \| missing terms: Town, Centre, Alton, United, Kingdom | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-4-31b-it-4bit` | `model_shortcoming` | missing terms: Centre, United, Kingdom \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `qnguyen3/nanoLLaVA` | `cutoff` | hit token cap (500) \| output/prompt=109.17% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | hit token cap (500) \| output/prompt=29.99% \| nontext prompt burden=77% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/pixtral-12b-8bit` | `clean` | nontext prompt burden=88% \| missing terms: Town, Centre, Alton, United, Kingdom \| keywords=19 | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-3n-E2B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=69.54% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | hit token cap (500) \| output/prompt=34.46% \| nontext prompt burden=73% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | hit token cap (500) \| output/prompt=29.99% \| nontext prompt burden=77% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: United, Kingdom | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `model_shortcoming` | missing terms: United, Kingdom \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-3n-E4B-it-bf16` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: Centre, Alton, United, Kingdom | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | hit token cap (500) \| output/prompt=34.46% \| nontext prompt burden=73% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | hit token cap (500) \| output/prompt=34.46% \| nontext prompt burden=73% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `cutoff` | hit token cap (500) \| output/prompt=39.53% \| nontext prompt burden=69% \| missing terms: Town, Centre, Alton, United, Kingdom | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=39.53%. |
| `microsoft/Phi-3.5-vision-instruct` | `cutoff` | hit token cap (500) \| output/prompt=39.53% \| nontext prompt burden=69% \| missing terms: Town, Centre, Alton, United, Kingdom | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=39.53%. |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `cutoff` | hit token cap (500) \| output/prompt=10.97% \| nontext prompt burden=92% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | hit token cap (500) \| output/prompt=14.92% \| nontext prompt burden=88% \| missing sections: title, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, keywords remained incomplete. |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `cutoff` | hit token cap (500) \| output/prompt=7.61% \| nontext prompt burden=94% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `cutoff` | hit token cap (500) \| output/prompt=18.24% \| nontext prompt burden=86% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/InternVL3-8B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=22.31% \| nontext prompt burden=83% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/GLM-4.6V-Flash-6bit` | `cutoff` | hit token cap (500) \| output/prompt=7.61% \| nontext prompt burden=94% \| missing sections: title, description | Raise the token cap or trim prompt burden first; generation hit the limit while title, description remained incomplete. |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff` | hit token cap (500) \| output/prompt=28.26% \| nontext prompt burden=78% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Molmo-7B-D-0924-8bit` | `cutoff` | hit token cap (500) \| output/prompt=30.40% \| nontext prompt burden=76% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `cutoff` | hit token cap (500) \| output/prompt=33.69% \| nontext prompt burden=74% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/GLM-4.6V-nvfp4` | `cutoff` | hit token cap (500) \| output/prompt=7.61% \| nontext prompt burden=94% \| missing sections: title | Raise the token cap or trim prompt burden first; generation hit the limit while title remained incomplete. |
| `mlx-community/pixtral-12b-bf16` | `cutoff` | hit token cap (500) \| output/prompt=15.33% \| nontext prompt burden=88% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Molmo-7B-D-0924-bf16` | `cutoff` | hit token cap (500) \| output/prompt=30.40% \| nontext prompt burden=76% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-4-31b-bf16` | `cutoff` | hit token cap (500) \| output/prompt=69.35% \| missing sections: title, description, keywords \| nonvisual metadata reused | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-27B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete. |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `cutoff` | hit token cap (500) \| output/prompt=116.28% \| missing sections: title, description, keywords \| missing terms: Centre, United, Kingdom | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=98% \| missing sections: description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete. |

### `model-config`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | processor error \| model config processor load processor | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=74% \| missing sections: title, description, keywords \| missing terms: Town, Centre, Alton, United, Kingdom | Check chat-template and EOS defaults first; the output shape is not matching the requested contract. |

## Model Verdicts

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|im_end|&gt; appeared in generated text. | keywords=36 | context echo=100% | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=36
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=387 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/MolmoPoint-8B-fp16`

- **Verdict:** harness | user=avoid
- **Why:** processor error | model config processor load processor
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=model-config | package=model-config | stage=Processor Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|im_end|&gt; appeared in generated text. | keywords=36 | context echo=100% | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=36
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=387 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|im_end|&gt; appeared in generated text. | keywords=37 | context echo=100% | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=37
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=387 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|im_end|&gt; appeared in generated text. | keywords=36 | context echo=100% | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=36
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=387 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: keywords | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=458 | text_est=387 | nontext_est=71 | gen=38 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=711 | text_est=387 | nontext_est=324 | gen=49 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=462 | text_est=387 | nontext_est=75 | gen=34 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=87% | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** description sentences=3
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3067 | text_est=387 | nontext_est=2680 | gen=131 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom | nonvisual metadata reused
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1484 | text_est=387 | nontext_est=1097 | gen=23 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=95.97% | missing terms: Town, Centre, Alton, United, Kingdom | keyword duplication=85%
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** title words=19 | description sentences=5 | keywords=125 | keyword duplication=0.85
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=521 | text_est=387 | nontext_est=134 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=95.97%.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=74% | missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1484 | text_est=387 | nontext_est=1097 | gen=12 | max=500 | stop=completed
- **Next action:** Check chat-template and EOS defaults first; the output shape is not matching the requested contract.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=83% | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** title words=2
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2241 | text_est=387 | nontext_est=1854 | gen=70 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: United, Kingdom | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=728 | text_est=387 | nontext_est=341 | gen=88 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=85% | missing sections: title, description, keywords | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2668 | text_est=387 | nontext_est=2281 | gen=109 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=87% | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** title words=12
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3068 | text_est=387 | nontext_est=2681 | gen=109 | max=500 | stop=completed
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

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=87% | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** ok
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3068 | text_est=387 | nontext_est=2681 | gen=126 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-4-31b-it-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: Centre, United, Kingdom | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=733 | text_est=387 | nontext_est=346 | gen=87 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

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
- **Why:** hit token cap (500) | output/prompt=29.99% | nontext prompt burden=77% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1667 | text_est=387 | nontext_est=1280 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=88% | missing terms: Town, Centre, Alton, United, Kingdom | keywords=19
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** description sentences=3 | keywords=19
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3261 | text_est=387 | nontext_est=2874 | gen=106 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=69.54% | missing sections: title, description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=719 | text_est=387 | nontext_est=332 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=34.46% | nontext prompt burden=73% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1451 | text_est=387 | nontext_est=1064 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 53 occurrences). | nontext prompt burden=85% | missing sections: description, keywords | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: description, keywords | title words=56
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2569 | text_est=387 | nontext_est=2182 | gen=77 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=29.99% | nontext prompt burden=77% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1667 | text_est=387 | nontext_est=1280 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: United, Kingdom
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=429 | text_est=387 | nontext_est=42 | gen=90 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: United, Kingdom | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=728 | text_est=387 | nontext_est=341 | gen=99 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: Centre, Alton, United, Kingdom
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=727 | text_est=387 | nontext_est=340 | gen=328 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=34.46% | nontext prompt burden=73% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1451 | text_est=387 | nontext_est=1064 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=34.46% | nontext prompt burden=73% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1451 | text_est=387 | nontext_est=1064 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=39.53% | nontext prompt burden=69% | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** title words=3 | keywords=92 | keyword duplication=0.73
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1265 | text_est=387 | nontext_est=878 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=39.53%.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=39.53% | nontext prompt burden=69% | missing terms: Town, Centre, Alton, United, Kingdom
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** title words=3 | keywords=92 | keyword duplication=0.73
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1265 | text_est=387 | nontext_est=878 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=39.53%.

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
- **Why:** hit token cap (500) | output/prompt=14.92% | nontext prompt burden=88% | missing sections: title, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3352 | text_est=387 | nontext_est=2965 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, keywords remained incomplete.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.61% | nontext prompt burden=94% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6574 | text_est=387 | nontext_est=6187 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=18.24% | nontext prompt burden=86% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2741 | text_est=387 | nontext_est=2354 | gen=500 | max=500 | stop=completed
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

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.61% | nontext prompt burden=94% | missing sections: title, description
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description | keywords=45 | keyword duplication=0.56
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6574 | text_est=387 | nontext_est=6187 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description remained incomplete.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=28.26% | nontext prompt burden=78% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1769 | text_est=387 | nontext_est=1382 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=30.40% | nontext prompt burden=76% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1645 | text_est=387 | nontext_est=1258 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=33.69% | nontext prompt burden=74% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1484 | text_est=387 | nontext_est=1097 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.61% | nontext prompt burden=94% | missing sections: title
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title | description sentences=5 | keywords=37 | keyword duplication=0.41
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6574 | text_est=387 | nontext_est=6187 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title remained incomplete.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=15.33% | nontext prompt burden=88% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3261 | text_est=387 | nontext_est=2874 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=30.40% | nontext prompt burden=76% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Town, Centre, Alton, United, Kingdom
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1645 | text_est=387 | nontext_est=1258 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16700 | text_est=387 | nontext_est=16313 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16700 | text_est=387 | nontext_est=16313 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16700 | text_est=387 | nontext_est=16313 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% | missing sections: title, description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16700 | text_est=387 | nontext_est=16313 | gen=500 | max=500 | stop=completed
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
- **Why:** hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% | missing sections: description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=51
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16700 | text_est=387 | nontext_est=16313 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete.

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

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% | missing sections: description, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=17
- **Utility:** user=avoid | improves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16700 | text_est=387 | nontext_est=16313 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete.
