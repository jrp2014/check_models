# Automated Review Digest

_Generated on 2026-04-06 23:33:33 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [check_models.log](check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🏆 A (88/100) | Δ+12 | 172.1 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (88/100) | Δ+12 | 61.1 tps
- `mlx-community/Molmo-7B-D-0924-8bit`: ✅ B (78/100) | Δ+2 | 52.2 tps

### Watchlist

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (1/100) | Δ-75 | 232.9 tps | context ignored, harness, long context
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (4/100) | Δ-72 | 31.0 tps | harness, missing sections
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) | Δ-65 | 69.8 tps | context ignored, harness, long context
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (18/100) | Δ-58 | 22.1 tps | context ignored, harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (22/100) | Δ-54 | 5.7 tps | context ignored, harness, missing sections

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | improves trusted hints | nontext prompt burden=90% \| missing terms: flies, low, style, bird, soaring |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `clean` | preserves trusted hints | missing terms: Japanese, style, garden, soaring, above |
| `mlx-community/gemma-4-31b-it-4bit` | `clean` | preserves trusted hints | missing terms: tranquil, Japanese, style, mid, soaring |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | preserves trusted hints | nontext prompt burden=90% \| missing terms: flies, low, tranquil, soaring, above |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | improves trusted hints | nontext prompt burden=90% \| missing terms: flies, low, style, soaring, water's |
| `mlx-community/pixtral-12b-8bit` | `clean` | preserves trusted hints | nontext prompt burden=91% \| context echo=54% |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `clean` | preserves trusted hints | missing terms: tranquil, Japanese, style, garden, soaring |

### `caveat`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `context_budget` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | Output was a short generic filler response (about 8 tokens). \| At long prompt length (3482 tokens), output stayed unusually short (8 tokens; ratio 0.2%). \| output/prompt=0.23% \| nontext prompt burden=88% |
| `mlx-community/InternVL3-14B-8bit` | `clean` | ignores trusted hints \| missing terms: heron, flies, low, over, tranquil | nontext prompt burden=86% \| missing terms: heron, flies, low, over, tranquil |
| `mlx-community/X-Reasoner-7B-8bit` | `clean` | preserves trusted hints | nontext prompt burden=97% \| missing terms: flies, low, over, water's, surface |
| `mlx-community/Molmo-7B-D-0924-8bit` | `clean` | preserves trusted hints | nontext prompt burden=75% |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | ignores trusted hints \| missing terms: grey, heron, flies, low, tranquil | Output appears truncated to about 6 tokens. \| At long prompt length (16752 tokens), output stayed unusually short (6 tokens; ratio 0.0%). \| output/prompt=0.04% \| nontext prompt burden=97% |

### `avoid`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | not evaluated | processor error \| model config processor load processor |
| `mlx-community/nanoLLaVA-1.5-4bit` | `model_shortcoming` | preserves trusted hints | missing sections: keywords \| context echo=73% |
| `mlx-community/FastVLM-0.5B-bf16` | `model_shortcoming` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over |
| `mlx-community/LFM2-VL-1.6B-8bit` | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused | missing sections: title, description, keywords \| missing terms: flies \| nonvisual metadata reused |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused | missing terms: flies, low \| nonvisual metadata reused |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `harness` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=72% \| missing terms: grey, heron, flies, low, over |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `model_shortcoming` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | nontext prompt burden=72% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=72% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=82.64% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over |
| `mlx-community/gemma-3n-E2B-4bit` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=66.14% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over |
| `qnguyen3/nanoLLaVA` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=101.63% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=24.17% \| nontext prompt burden=80% \| missing sections: title, description, keywords |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=24.17% \| nontext prompt burden=80% \| missing sections: title, description, keywords |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=33.49% \| nontext prompt burden=72% \| missing sections: title, description, keywords |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=33.49% \| nontext prompt burden=72% \| missing sections: title, description, keywords |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=33.49% \| nontext prompt burden=72% \| missing sections: title, description, keywords |
| `mlx-community/gemma-3n-E4B-it-bf16` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | missing terms: flies, low, above, water's, surface \| nonvisual metadata reused |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | preserves trusted hints | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 48 occurrences). \| nontext prompt burden=88% \| missing sections: description, keywords \| missing terms: flies, low, bird, soaring, above |
| `microsoft/Phi-3.5-vision-instruct` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=38.11% \| nontext prompt burden=68% \| missing terms: flies, low, soaring, above, water's |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=38.11% \| nontext prompt burden=68% \| missing terms: flies, low, soaring, above, water's |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=7.64% \| nontext prompt burden=94% \| missing sections: title, description, keywords |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=10.89% \| nontext prompt burden=91% \| missing sections: title, description, keywords |
| `mlx-community/GLM-4.6V-Flash-6bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=7.64% \| nontext prompt burden=94% \| missing sections: title, description, keywords |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=10.57% \| nontext prompt burden=91% \| missing sections: title, description, keywords |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=27.95% \| nontext prompt burden=76% \| missing sections: title, description, keywords |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=14.35% \| nontext prompt burden=88% \| missing sections: title, description, keywords |
| `mlx-community/InternVL3-8B-bf16` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=16.43% \| nontext prompt burden=86% \| missing sections: title, description, keywords |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | ignores trusted hints \| missing terms: heron, flies, low, over, tranquil | At long prompt length (16741 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=97% |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | At long prompt length (16743 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=97% |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=108.46% \| missing sections: title, description, keywords \| missing terms: flies, low, bird, soaring, above |
| `mlx-community/GLM-4.6V-nvfp4` | `cutoff` | improves trusted hints | hit token cap (500) \| output/prompt=7.64% \| nontext prompt burden=94% \| missing sections: title, description |
| `mlx-community/pixtral-12b-bf16` | `cutoff` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | hit token cap (500) \| output/prompt=10.77% \| nontext prompt burden=91% \| missing sections: title, description, keywords |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `model_shortcoming` | preserves trusted hints | missing sections: title, description, keywords \| missing terms: grey, flies, style, bird, mid |
| `mlx-community/Molmo-7B-D-0924-bf16` | `model_shortcoming` | ignores trusted hints \| missing terms: grey, heron, flies, low, over | nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | degrades trusted hints | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-27B-4bit` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: keywords |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: keywords |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `mlx`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `context_budget` | Output was a short generic filler response (about 8 tokens). \| At long prompt length (3482 tokens), output stayed unusually short (8 tokens; ratio 0.2%). \| output/prompt=0.23% \| nontext prompt burden=88% | Treat this as a prompt-budget issue first; nontext prompt burden is 88% and the output stays weak under that load. |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | At long prompt length (16741 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=97% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.99%. |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | At long prompt length (16743 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=97% | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | Output appears truncated to about 6 tokens. \| At long prompt length (16752 tokens), output stayed unusually short (6 tokens; ratio 0.0%). \| output/prompt=0.04% \| nontext prompt burden=97% | Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load. |

### `mlx-vlm`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 48 occurrences). \| nontext prompt burden=88% \| missing sections: description, keywords \| missing terms: flies, low, bird, soaring, above | Inspect decode cleanup; tokenizer markers are leaking into user-facing text. |

### `model`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/nanoLLaVA-1.5-4bit` | `model_shortcoming` | missing sections: keywords \| context echo=73% | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/FastVLM-0.5B-bf16` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/LFM2-VL-1.6B-8bit` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: flies \| nonvisual metadata reused | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `model_shortcoming` | missing terms: flies, low \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `model_shortcoming` | nontext prompt burden=72% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | nontext prompt burden=90% \| missing terms: flies, low, style, bird, soaring | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `clean` | missing terms: Japanese, style, garden, soaring, above | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `cutoff` | hit token cap (500) \| output/prompt=82.64% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-3n-E2B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=66.14% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-4-31b-it-4bit` | `clean` | missing terms: tranquil, Japanese, style, mid, soaring | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `qnguyen3/nanoLLaVA` | `cutoff` | hit token cap (500) \| output/prompt=101.63% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | hit token cap (500) \| output/prompt=24.17% \| nontext prompt burden=80% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/InternVL3-14B-8bit` | `clean` | nontext prompt burden=86% \| missing terms: heron, flies, low, over, tranquil | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | hit token cap (500) \| output/prompt=24.17% \| nontext prompt burden=80% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | hit token cap (500) \| output/prompt=33.49% \| nontext prompt burden=72% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | nontext prompt burden=90% \| missing terms: flies, low, tranquil, soaring, above | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | nontext prompt burden=90% \| missing terms: flies, low, style, soaring, water's | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/pixtral-12b-8bit` | `clean` | nontext prompt burden=91% \| context echo=54% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | hit token cap (500) \| output/prompt=33.49% \| nontext prompt burden=72% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `clean` | missing terms: tranquil, Japanese, style, garden, soaring | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | hit token cap (500) \| output/prompt=33.49% \| nontext prompt burden=72% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-3n-E4B-it-bf16` | `model_shortcoming` | missing terms: flies, low, above, water's, surface \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `microsoft/Phi-3.5-vision-instruct` | `cutoff` | hit token cap (500) \| output/prompt=38.11% \| nontext prompt burden=68% \| missing terms: flies, low, soaring, above, water's | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=38.11%. |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `cutoff` | hit token cap (500) \| output/prompt=38.11% \| nontext prompt burden=68% \| missing terms: flies, low, soaring, above, water's | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=38.11%. |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `cutoff` | hit token cap (500) \| output/prompt=7.64% \| nontext prompt burden=94% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `cutoff` | hit token cap (500) \| output/prompt=10.89% \| nontext prompt burden=91% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/GLM-4.6V-Flash-6bit` | `cutoff` | hit token cap (500) \| output/prompt=7.64% \| nontext prompt burden=94% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | hit token cap (500) \| output/prompt=10.57% \| nontext prompt burden=91% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff` | hit token cap (500) \| output/prompt=27.95% \| nontext prompt burden=76% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `cutoff` | hit token cap (500) \| output/prompt=14.35% \| nontext prompt burden=88% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/X-Reasoner-7B-8bit` | `clean` | nontext prompt burden=97% \| missing terms: flies, low, over, water's, surface | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/InternVL3-8B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=16.43% \| nontext prompt burden=86% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Molmo-7B-D-0924-8bit` | `clean` | nontext prompt burden=75% | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `cutoff` | hit token cap (500) \| output/prompt=108.46% \| missing sections: title, description, keywords \| missing terms: flies, low, bird, soaring, above | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/GLM-4.6V-nvfp4` | `cutoff` | hit token cap (500) \| output/prompt=7.64% \| nontext prompt burden=94% \| missing sections: title, description | Raise the token cap or trim prompt burden first; generation hit the limit while title, description remained incomplete. |
| `mlx-community/pixtral-12b-bf16` | `cutoff` | hit token cap (500) \| output/prompt=10.77% \| nontext prompt burden=91% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: grey, flies, style, bird, mid | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/Molmo-7B-D-0924-bf16` | `model_shortcoming` | nontext prompt burden=75% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-27B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: keywords | Raise the token cap or trim prompt burden first; generation hit the limit while keywords remained incomplete. |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: keywords | Raise the token cap or trim prompt burden first; generation hit the limit while keywords remained incomplete. |

### `model-config`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | processor error \| model config processor load processor | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `harness` | Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=72% \| missing terms: grey, heron, flies, low, over | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=72% \| missing sections: title, description, keywords \| missing terms: grey, heron, flies, low, over | Check chat-template and EOS defaults first; the output shape is not matching the requested contract. |

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

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: keywords | context echo=73%
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: keywords | title words=11
- **Utility:** user=avoid | preserves trusted hints | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=492 | text_est=424 | nontext_est=68 | gen=68 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: grey, heron, flies, low, over
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=496 | text_est=424 | nontext_est=72 | gen=22 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: flies | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=745 | text_est=424 | nontext_est=321 | gen=120 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: flies, low | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=553 | text_est=424 | nontext_est=129 | gen=125 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=72% | missing terms: grey, heron, flies, low, over
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1521 | text_est=424 | nontext_est=1097 | gen=8 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=72% | missing sections: title, description, keywords | missing terms: grey, heron, flies, low, over
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1521 | text_est=424 | nontext_est=1097 | gen=25 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** context_budget | user=caveat
- **Why:** Output was a short generic filler response (about 8 tokens). | At long prompt length (3482 tokens), output stayed unusually short (8 tokens; ratio 0.2%). | output/prompt=0.23% | nontext prompt burden=88%
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** ok
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=3482 | text_est=424 | nontext_est=3058 | gen=8 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 88% and the output stays weak under that load.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=90% | missing terms: flies, low, style, bird, soaring
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4093 | text_est=424 | nontext_est=3669 | gen=100 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=72% | missing sections: title, description, keywords | missing terms: grey, heron, flies, low, over
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1521 | text_est=424 | nontext_est=1097 | gen=12 | max=500 | stop=completed
- **Next action:** Check chat-template and EOS defaults first; the output shape is not matching the requested contract.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** clean | user=recommended
- **Why:** missing terms: Japanese, style, garden, soaring, above
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=765 | text_est=424 | nontext_est=341 | gen=93 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=82.64% | missing sections: title, description, keywords | missing terms: grey, heron, flies, low, over
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=605 | text_est=424 | nontext_est=181 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=66.14% | missing sections: title, description, keywords | missing terms: grey, heron, flies, low, over
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=756 | text_est=424 | nontext_est=332 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/gemma-4-31b-it-4bit`

- **Verdict:** clean | user=recommended
- **Why:** missing terms: tranquil, Japanese, style, mid, soaring
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=767 | text_est=424 | nontext_est=343 | gen=79 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=101.63% | missing sections: title, description, keywords | missing terms: grey, heron, flies, low, over
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=492 | text_est=424 | nontext_est=68 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=24.17% | nontext prompt burden=80% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2069 | text_est=424 | nontext_est=1645 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=86% | missing terms: heron, flies, low, over, tranquil
- **Trusted hints:** ignores trusted hints | missing terms: heron, flies, low, over, tranquil
- **Contract:** title words=2
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3043 | text_est=424 | nontext_est=2619 | gen=80 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=24.17% | nontext prompt burden=80% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2069 | text_est=424 | nontext_est=1645 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=33.49% | nontext prompt burden=72% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1493 | text_est=424 | nontext_est=1069 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=90% | missing terms: flies, low, tranquil, soaring, above
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4094 | text_est=424 | nontext_est=3670 | gen=109 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=90% | missing terms: flies, low, style, soaring, water's
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4094 | text_est=424 | nontext_est=3670 | gen=107 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=91% | context echo=54%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4641 | text_est=424 | nontext_est=4217 | gen=99 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=33.49% | nontext prompt burden=72% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1493 | text_est=424 | nontext_est=1069 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** clean | user=recommended
- **Why:** missing terms: tranquil, Japanese, style, garden, soaring
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=765 | text_est=424 | nontext_est=341 | gen=98 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=33.49% | nontext prompt burden=72% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1493 | text_est=424 | nontext_est=1069 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: flies, low, above, water's, surface | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** description sentences=9
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=764 | text_est=424 | nontext_est=340 | gen=373 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 48 occurrences). | nontext prompt burden=88% | missing sections: description, keywords | missing terms: flies, low, bird, soaring, above
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: description, keywords | title words=53
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=3598 | text_est=424 | nontext_est=3174 | gen=82 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=38.11% | nontext prompt burden=68% | missing terms: flies, low, soaring, above, water's
- **Trusted hints:** preserves trusted hints
- **Contract:** keywords=180 | keyword duplication=0.84
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1312 | text_est=424 | nontext_est=888 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=38.11%.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=38.11% | nontext prompt burden=68% | missing terms: flies, low, soaring, above, water's
- **Trusted hints:** preserves trusted hints
- **Contract:** keywords=180 | keyword duplication=0.84
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1312 | text_est=424 | nontext_est=888 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=38.11%.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.64% | nontext prompt burden=94% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6545 | text_est=424 | nontext_est=6121 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=10.89% | nontext prompt burden=91% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4593 | text_est=424 | nontext_est=4169 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.64% | nontext prompt burden=94% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6545 | text_est=424 | nontext_est=6121 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=10.57% | nontext prompt burden=91% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4732 | text_est=424 | nontext_est=4308 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=27.95% | nontext prompt burden=76% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1789 | text_est=424 | nontext_est=1365 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=14.35% | nontext prompt burden=88% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3485 | text_est=424 | nontext_est=3061 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=97% | missing terms: flies, low, over, water's, surface
- **Trusted hints:** preserves trusted hints
- **Contract:** title words=4
- **Utility:** user=caveat | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16752 | text_est=424 | nontext_est=16328 | gen=110 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=16.43% | nontext prompt burden=86% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3043 | text_est=424 | nontext_est=2619 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** At long prompt length (16741 tokens), output became repetitive. | hit token cap (500) | output/prompt=2.99% | nontext prompt burden=97%
- **Trusted hints:** ignores trusted hints | missing terms: heron, flies, low, over, tranquil
- **Contract:** title words=19 | keywords=118 | keyword duplication=0.83
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16741 | text_est=424 | nontext_est=16317 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.99%.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** At long prompt length (16743 tokens), output became repetitive. | hit token cap (500) | output/prompt=2.99% | nontext prompt burden=97%
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16743 | text_est=424 | nontext_est=16319 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=75%
- **Trusted hints:** preserves trusted hints
- **Contract:** description sentences=7
- **Utility:** user=caveat | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1671 | text_est=424 | nontext_est=1247 | gen=326 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=108.46% | missing sections: title, description, keywords | missing terms: flies, low, bird, soaring, above
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=461 | text_est=424 | nontext_est=37 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.64% | nontext prompt burden=94% | missing sections: title, description
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description | keywords=55 | keyword duplication=0.49
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6545 | text_est=424 | nontext_est=6121 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description remained incomplete.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=10.77% | nontext prompt burden=91% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4641 | text_est=424 | nontext_est=4217 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: grey, flies, style, bird, mid
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=462 | text_est=424 | nontext_est=38 | gen=171 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=75% | missing sections: title, description, keywords | missing terms: grey, heron, flies, low, over
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, over
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1671 | text_est=424 | nontext_est=1247 | gen=258 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16767 | text_est=424 | nontext_est=16343 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** degrades trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16767 | text_est=424 | nontext_est=16343 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing sections: description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=30
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16767 | text_est=424 | nontext_est=16343 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** Output appears truncated to about 6 tokens. | At long prompt length (16752 tokens), output stayed unusually short (6 tokens; ratio 0.0%). | output/prompt=0.04% | nontext prompt burden=97%
- **Trusted hints:** ignores trusted hints | missing terms: grey, heron, flies, low, tranquil
- **Contract:** ok
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16752 | text_est=424 | nontext_est=16328 | gen=6 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16767 | text_est=424 | nontext_est=16343 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing sections: keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords | title words=41
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16767 | text_est=424 | nontext_est=16343 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while keywords remained incomplete.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing sections: keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords | title words=26 | description sentences=4
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16767 | text_est=424 | nontext_est=16343 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while keywords remained incomplete.
