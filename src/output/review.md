# Automated Review Digest

_Generated on 2026-04-11 00:42:19 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [check_models.log](check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (90/100) | Δ+16 | 66.3 tps
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: ✅ B (76/100) | Δ+1 | 185.5 tps

### Watchlist

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) | Δ-74 | 282.4 tps | context ignored, harness, long context
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) | Δ-68 | 32.0 tps | harness, missing sections
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) | Δ-64 | 70.0 tps | context ignored, harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (17/100) | Δ-57 | 5.9 tps | context ignored, harness
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (18/100) | Δ-56 | 22.2 tps | context ignored, harness

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | preserves trusted hints | nontext prompt burden=86% \| missing terms: style, azumaya, extends, rainy, day |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `clean` | preserves trusted hints | nontext prompt burden=67% \| missing terms: surrounding, area, features \| context echo=96% |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | preserves trusted hints | nontext prompt burden=86% \| missing terms: style, azumaya, extends, rainy, dark |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `clean` | preserves trusted hints | missing terms: traditional, Japanese, style, azumaya, Its |
| `mlx-community/gemma-4-31b-it-4bit` | `clean` | preserves trusted hints | missing terms: traditional, azumaya, tranquil, rainy, day |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `clean` | preserves trusted hints | missing terms: traditional, Japanese, style, azumaya, tranquil |

### `caveat`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `clean` | preserves trusted hints | missing terms: azumaya \| keywords=8 \| context echo=48% |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | improves trusted hints | nontext prompt burden=86% \| missing terms: traditional, Japanese, style, azumaya, extends |
| `mlx-community/InternVL3-14B-8bit` | `clean` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | nontext prompt burden=81% \| missing terms: traditional, wooden, Japanese, style, gazebo |
| `mlx-community/pixtral-12b-8bit` | `clean` | preserves trusted hints | nontext prompt burden=87% \| missing terms: azumaya \| context echo=51% |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | Output appears truncated to about 3 tokens. \| At long prompt length (16738 tokens), output stayed unusually short (3 tokens; ratio 0.0%). \| output/prompt=0.02% \| nontext prompt burden=97% |

### `avoid`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | not evaluated | processor error \| model config processor load processor |
| `mlx-community/nanoLLaVA-1.5-4bit` | `model_shortcoming` | preserves trusted hints | missing sections: keywords \| missing terms: wooden, style, azumaya \| context echo=98% |
| `mlx-community/LFM2-VL-1.6B-8bit` | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused | missing sections: title, description, keywords \| missing terms: azumaya, extends, over, rainy, day \| nonvisual metadata reused |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `harness` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=71% \| missing terms: traditional, wooden, Japanese, style, gazebo |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `model_shortcoming` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: traditional, wooden, Japanese, style, gazebo |
| `mlx-community/FastVLM-0.5B-bf16` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo \| nonvisual metadata reused | hit token cap (500) \| output/prompt=97.28% \| missing sections: title, description, keywords \| missing terms: traditional, wooden, Japanese, style, gazebo |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `harness` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | Output was a short generic filler response (about 8 tokens). \| nontext prompt burden=84% \| missing terms: traditional, wooden, Japanese, style, gazebo |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=71% \| missing terms: traditional, wooden, Japanese, style, gazebo |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | hit token cap (500) \| output/prompt=80.26% \| missing sections: title, description, keywords \| missing terms: traditional, wooden, Japanese, style, gazebo |
| `mlx-community/gemma-3n-E2B-4bit` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo \| nonvisual metadata reused | hit token cap (500) \| output/prompt=64.85% \| missing sections: title, description, keywords \| missing terms: traditional, wooden, Japanese, style, gazebo |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | hit token cap (500) \| output/prompt=29.02% \| nontext prompt burden=75% \| missing sections: title, description, keywords |
| `qnguyen3/nanoLLaVA` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | hit token cap (500) \| output/prompt=98.04% \| missing sections: title, description, keywords \| missing terms: traditional, wooden, Japanese, style, gazebo |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | hit token cap (500) \| output/prompt=29.02% \| nontext prompt burden=75% \| missing sections: title, description, keywords |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo \| nonvisual metadata reused | hit token cap (500) \| output/prompt=33.33% \| nontext prompt burden=71% \| missing sections: title, description, keywords |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | ignores trusted hints \| missing terms: Japanese, style, gazebo, azumaya, extends | hit token cap (500) \| output/prompt=33.33% \| nontext prompt burden=71% \| missing sections: title, description, keywords |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `model_shortcoming` | preserves trusted hints | missing sections: keywords \| missing terms: extends, Its |
| `mlx-community/gemma-3n-E4B-it-bf16` | `model_shortcoming` | improves trusted hints | missing sections: title, description, keywords \| missing terms: extends, rainy, visible, rippling, contemplative |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | preserves trusted hints | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 57 occurrences). \| nontext prompt burden=83% \| missing sections: description, keywords \| missing terms: Japanese, style, rainy, day, dark |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | hit token cap (500) \| output/prompt=33.33% \| nontext prompt burden=71% \| missing sections: title, description, keywords |
| `microsoft/Phi-3.5-vision-instruct` | `harness` | preserves trusted hints | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=67% |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | hit token cap (500) \| output/prompt=10.85% \| nontext prompt burden=90% \| missing sections: title, description, keywords |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=7.55% \| nontext prompt burden=93% \| missing sections: title, description, keywords |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=14.67% \| nontext prompt burden=87% \| missing sections: title, keywords |
| `mlx-community/GLM-4.6V-Flash-6bit` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=7.55% \| nontext prompt burden=93% \| missing sections: title, description, keywords |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=27.31% \| nontext prompt burden=76% \| missing sections: title, description, keywords |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | hit token cap (500) \| output/prompt=17.90% \| nontext prompt burden=84% \| missing sections: title, description, keywords |
| `mlx-community/InternVL3-8B-bf16` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | hit token cap (500) \| output/prompt=21.81% \| nontext prompt burden=81% \| missing sections: title, description, keywords |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | preserves trusted hints | At long prompt length (16727 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=97% |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | ignores trusted hints \| missing terms: traditional, Japanese, style, gazebo, azumaya | At long prompt length (16729 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=97% |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `model_shortcoming` | preserves trusted hints | missing sections: title, description, keywords \| missing terms: wooden, extends, Its, contemplative, features |
| `mlx-community/Molmo-7B-D-0924-8bit` | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused | nontext prompt burden=74% \| nonvisual metadata reused |
| `mlx-community/X-Reasoner-7B-8bit` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | At long prompt length (16738 tokens), output may stop following prompt/image context. \| hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=97% |
| `mlx-community/GLM-4.6V-nvfp4` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=7.55% \| nontext prompt burden=93% \| missing sections: title, description |
| `mlx-community/pixtral-12b-bf16` | `cutoff` | degrades trusted hints | hit token cap (500) \| output/prompt=15.07% \| nontext prompt burden=87% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | improves trusted hints | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing terms: azumaya, extends, rainy, day, Its |
| `mlx-community/Molmo-7B-D-0924-bf16` | `cutoff` | ignores trusted hints \| missing terms: traditional, wooden, Japanese, style, gazebo | hit token cap (500) \| output/prompt=29.46% \| nontext prompt burden=74% \| missing sections: title, description, keywords |
| `mlx-community/gemma-4-31b-bf16` | `cutoff` | preserves trusted hints | hit token cap (500) \| output/prompt=64.68% \| missing sections: title, description, keywords \| missing terms: wooden, style, extends, over, Its |
| `mlx-community/Qwen3.5-27B-4bit` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `mlx`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | At long prompt length (16727 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=97% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.99%. |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | At long prompt length (16729 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=97% | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/X-Reasoner-7B-8bit` | `cutoff` | At long prompt length (16738 tokens), output may stop following prompt/image context. \| hit token cap (500) \| output/prompt=2.99% \| nontext prompt burden=97% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.99%. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | Output appears truncated to about 3 tokens. \| At long prompt length (16738 tokens), output stayed unusually short (3 tokens; ratio 0.0%). \| output/prompt=0.02% \| nontext prompt burden=97% | Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load. |

### `mlx-vlm`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 57 occurrences). \| nontext prompt burden=83% \| missing sections: description, keywords \| missing terms: Japanese, style, rainy, day, dark | Inspect decode cleanup; tokenizer markers are leaking into user-facing text. |
| `microsoft/Phi-3.5-vision-instruct` | `harness` | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=67% | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/nanoLLaVA-1.5-4bit` | `model_shortcoming` | missing sections: keywords \| missing terms: wooden, style, azumaya \| context echo=98% | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/LFM2-VL-1.6B-8bit` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: azumaya, extends, over, rainy, day \| nonvisual metadata reused | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `clean` | missing terms: azumaya \| keywords=8 \| context echo=48% | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | nontext prompt burden=86% \| missing terms: style, azumaya, extends, rainy, day | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `model_shortcoming` | nontext prompt burden=71% \| missing sections: title, description, keywords \| missing terms: traditional, wooden, Japanese, style, gazebo | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/FastVLM-0.5B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=97.28% \| missing sections: title, description, keywords \| missing terms: traditional, wooden, Japanese, style, gazebo | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `clean` | nontext prompt burden=67% \| missing terms: surrounding, area, features \| context echo=96% | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | nontext prompt burden=86% \| missing terms: style, azumaya, extends, rainy, dark | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `clean` | missing terms: traditional, Japanese, style, azumaya, Its | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `cutoff` | hit token cap (500) \| output/prompt=80.26% \| missing sections: title, description, keywords \| missing terms: traditional, wooden, Japanese, style, gazebo | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-3n-E2B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=64.85% \| missing sections: title, description, keywords \| missing terms: traditional, wooden, Japanese, style, gazebo | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | nontext prompt burden=86% \| missing terms: traditional, Japanese, style, azumaya, extends | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | hit token cap (500) \| output/prompt=29.02% \| nontext prompt burden=75% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-4-31b-it-4bit` | `clean` | missing terms: traditional, azumaya, tranquil, rainy, day | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/InternVL3-14B-8bit` | `clean` | nontext prompt burden=81% \| missing terms: traditional, wooden, Japanese, style, gazebo | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `qnguyen3/nanoLLaVA` | `cutoff` | hit token cap (500) \| output/prompt=98.04% \| missing sections: title, description, keywords \| missing terms: traditional, wooden, Japanese, style, gazebo | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | hit token cap (500) \| output/prompt=29.02% \| nontext prompt burden=75% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/pixtral-12b-8bit` | `clean` | nontext prompt burden=87% \| missing terms: azumaya \| context echo=51% | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | hit token cap (500) \| output/prompt=33.33% \| nontext prompt burden=71% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | hit token cap (500) \| output/prompt=33.33% \| nontext prompt burden=71% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `model_shortcoming` | missing sections: keywords \| missing terms: extends, Its | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `clean` | missing terms: traditional, Japanese, style, azumaya, tranquil | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-3n-E4B-it-bf16` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: extends, rainy, visible, rippling, contemplative | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | hit token cap (500) \| output/prompt=33.33% \| nontext prompt burden=71% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `cutoff` | hit token cap (500) \| output/prompt=10.85% \| nontext prompt burden=90% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `cutoff` | hit token cap (500) \| output/prompt=7.55% \| nontext prompt burden=93% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | hit token cap (500) \| output/prompt=14.67% \| nontext prompt burden=87% \| missing sections: title, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, keywords remained incomplete. |
| `mlx-community/GLM-4.6V-Flash-6bit` | `cutoff` | hit token cap (500) \| output/prompt=7.55% \| nontext prompt burden=93% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff` | hit token cap (500) \| output/prompt=27.31% \| nontext prompt burden=76% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `cutoff` | hit token cap (500) \| output/prompt=17.90% \| nontext prompt burden=84% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/InternVL3-8B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=21.81% \| nontext prompt burden=81% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: wooden, extends, Its, contemplative, features | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/Molmo-7B-D-0924-8bit` | `model_shortcoming` | nontext prompt burden=74% \| nonvisual metadata reused | Treat as a model-quality limitation for this prompt and image. |
| `mlx-community/GLM-4.6V-nvfp4` | `cutoff` | hit token cap (500) \| output/prompt=7.55% \| nontext prompt burden=93% \| missing sections: title, description | Raise the token cap or trim prompt burden first; generation hit the limit while title, description remained incomplete. |
| `mlx-community/pixtral-12b-bf16` | `cutoff` | hit token cap (500) \| output/prompt=15.07% \| nontext prompt burden=87% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing terms: azumaya, extends, rainy, day, Its | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.98%. |
| `mlx-community/Molmo-7B-D-0924-bf16` | `cutoff` | hit token cap (500) \| output/prompt=29.46% \| nontext prompt burden=74% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-4-31b-bf16` | `cutoff` | hit token cap (500) \| output/prompt=64.68% \| missing sections: title, description, keywords \| missing terms: wooden, style, extends, over, Its | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-27B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | hit token cap (500) \| output/prompt=2.98% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |

### `model-config`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | processor error \| model config processor load processor | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `harness` | Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=71% \| missing terms: traditional, wooden, Japanese, style, gazebo | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `harness` | Output was a short generic filler response (about 8 tokens). \| nontext prompt burden=84% \| missing terms: traditional, wooden, Japanese, style, gazebo | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=71% \| missing terms: traditional, wooden, Japanese, style, gazebo | Inspect model repo config, chat template, and EOS settings. |

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
- **Why:** missing sections: keywords | missing terms: wooden, style, azumaya | context echo=98%
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: keywords | description sentences=3
- **Utility:** user=avoid | preserves trusted hints | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=510 | text_est=439 | nontext_est=71 | gen=74 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: azumaya, extends, over, rainy, day | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=767 | text_est=439 | nontext_est=328 | gen=118 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** clean | user=caveat
- **Why:** missing terms: azumaya | keywords=8 | context echo=48%
- **Trusted hints:** preserves trusted hints
- **Contract:** title words=19 | description sentences=3 | keywords=8
- **Utility:** user=caveat | preserves trusted hints | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=577 | text_est=439 | nontext_est=138 | gen=142 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=71% | missing terms: traditional, wooden, Japanese, style, gazebo
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1536 | text_est=439 | nontext_est=1097 | gen=8 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=86% | missing terms: style, azumaya, extends, rainy, day
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3123 | text_est=439 | nontext_est=2684 | gen=123 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=71% | missing sections: title, description, keywords | missing terms: traditional, wooden, Japanese, style, gazebo
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1536 | text_est=439 | nontext_est=1097 | gen=23 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=97.28% | missing sections: title, description, keywords | missing terms: traditional, wooden, Japanese, style, gazebo
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=514 | text_est=439 | nontext_est=75 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** harness | user=avoid
- **Why:** Output was a short generic filler response (about 8 tokens). | nontext prompt burden=84% | missing terms: traditional, wooden, Japanese, style, gazebo
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=2733 | text_est=439 | nontext_est=2294 | gen=8 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=67% | missing terms: surrounding, area, features | context echo=96%
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1335 | text_est=439 | nontext_est=896 | gen=138 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=71% | missing terms: traditional, wooden, Japanese, style, gazebo
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1536 | text_est=439 | nontext_est=1097 | gen=9 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=86% | missing terms: style, azumaya, extends, rainy, dark
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3124 | text_est=439 | nontext_est=2685 | gen=118 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** clean | user=recommended
- **Why:** missing terms: traditional, Japanese, style, azumaya, Its
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=780 | text_est=439 | nontext_est=341 | gen=94 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=80.26% | missing sections: title, description, keywords | missing terms: traditional, wooden, Japanese, style, gazebo
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=623 | text_est=439 | nontext_est=184 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=64.85% | missing sections: title, description, keywords | missing terms: traditional, wooden, Japanese, style, gazebo
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=771 | text_est=439 | nontext_est=332 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=86% | missing terms: traditional, Japanese, style, azumaya, extends
- **Trusted hints:** improves trusted hints
- **Contract:** description sentences=3
- **Utility:** user=caveat | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3124 | text_est=439 | nontext_est=2685 | gen=139 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=29.02% | nontext prompt burden=75% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1723 | text_est=439 | nontext_est=1284 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/gemma-4-31b-it-4bit`

- **Verdict:** clean | user=recommended
- **Why:** missing terms: traditional, azumaya, tranquil, rainy, day
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=786 | text_est=439 | nontext_est=347 | gen=86 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=81% | missing terms: traditional, wooden, Japanese, style, gazebo
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** ok
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2293 | text_est=439 | nontext_est=1854 | gen=93 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=98.04% | missing sections: title, description, keywords | missing terms: traditional, wooden, Japanese, style, gazebo
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=510 | text_est=439 | nontext_est=71 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=29.02% | nontext prompt burden=75% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1723 | text_est=439 | nontext_est=1284 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=87% | missing terms: azumaya | context echo=51%
- **Trusted hints:** preserves trusted hints
- **Contract:** title words=4 | description sentences=3
- **Utility:** user=caveat | preserves trusted hints | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3317 | text_est=439 | nontext_est=2878 | gen=112 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=33.33% | nontext prompt burden=71% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1500 | text_est=439 | nontext_est=1061 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=33.33% | nontext prompt burden=71% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: Japanese, style, gazebo, azumaya, extends
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1500 | text_est=439 | nontext_est=1061 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: keywords | missing terms: extends, Its
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: keywords | description sentences=3
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=481 | text_est=439 | nontext_est=42 | gen=88 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** clean | user=recommended
- **Why:** missing terms: traditional, Japanese, style, azumaya, tranquil
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=780 | text_est=439 | nontext_est=341 | gen=88 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: extends, rainy, visible, rippling, contemplative
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=779 | text_est=439 | nontext_est=340 | gen=306 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 57 occurrences). | nontext prompt burden=83% | missing sections: description, keywords | missing terms: Japanese, style, rainy, day, dark
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: description, keywords | title words=60
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2628 | text_est=439 | nontext_est=2189 | gen=103 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=33.33% | nontext prompt burden=71% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1500 | text_est=439 | nontext_est=1061 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;|end|&gt; appeared in generated text. | Special control token &lt;|endoftext|&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=67%
- **Trusted hints:** preserves trusted hints
- **Contract:** keywords=42
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1335 | text_est=439 | nontext_est=896 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=10.85% | nontext prompt burden=90% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4608 | text_est=439 | nontext_est=4169 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.55% | nontext prompt burden=93% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6625 | text_est=439 | nontext_est=6186 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=14.67% | nontext prompt burden=87% | missing sections: title, keywords
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, keywords | description sentences=6
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3408 | text_est=439 | nontext_est=2969 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, keywords remained incomplete.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.55% | nontext prompt burden=93% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6625 | text_est=439 | nontext_est=6186 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=27.31% | nontext prompt burden=76% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1831 | text_est=439 | nontext_est=1392 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=17.90% | nontext prompt burden=84% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2793 | text_est=439 | nontext_est=2354 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=21.81% | nontext prompt burden=81% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2293 | text_est=439 | nontext_est=1854 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** At long prompt length (16727 tokens), output became repetitive. | hit token cap (500) | output/prompt=2.99% | nontext prompt burden=97%
- **Trusted hints:** preserves trusted hints
- **Contract:** title words=36 | description sentences=3 | keywords=128 | keyword duplication=0.80
- **Utility:** user=avoid | preserves trusted hints | context echo
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16727 | text_est=439 | nontext_est=16288 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.99%.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** At long prompt length (16729 tokens), output became repetitive. | hit token cap (500) | output/prompt=2.99% | nontext prompt burden=97%
- **Trusted hints:** ignores trusted hints | missing terms: traditional, Japanese, style, gazebo, azumaya
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16729 | text_est=439 | nontext_est=16290 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: wooden, extends, Its, contemplative, features
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=482 | text_est=439 | nontext_est=43 | gen=104 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=74% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=11 | description sentences=3
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1697 | text_est=439 | nontext_est=1258 | gen=261 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** At long prompt length (16738 tokens), output may stop following prompt/image context. | hit token cap (500) | output/prompt=2.99% | nontext prompt burden=97%
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** description sentences=3 | keywords=73
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16738 | text_est=439 | nontext_est=16299 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.99%.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.55% | nontext prompt burden=93% | missing sections: title, description
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description | keywords=20
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6625 | text_est=439 | nontext_est=6186 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description remained incomplete.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=15.07% | nontext prompt burden=87% | missing sections: title, description, keywords
- **Trusted hints:** degrades trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3317 | text_est=439 | nontext_est=2878 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing sections: description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=16
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16753 | text_est=439 | nontext_est=16314 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16753 | text_est=439 | nontext_est=16314 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing sections: description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=34
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16753 | text_est=439 | nontext_est=16314 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while description, keywords remained incomplete.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** Output appears truncated to about 3 tokens. | At long prompt length (16738 tokens), output stayed unusually short (3 tokens; ratio 0.0%). | output/prompt=0.02% | nontext prompt burden=97%
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** ok
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16738 | text_est=439 | nontext_est=16299 | gen=3 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing terms: azumaya, extends, rainy, day, Its
- **Trusted hints:** improves trusted hints
- **Contract:** title words=41 | description sentences=5 | keywords=35 | keyword duplication=0.46
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16753 | text_est=439 | nontext_est=16314 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.98%.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=29.46% | nontext prompt burden=74% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: traditional, wooden, Japanese, style, gazebo
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1697 | text_est=439 | nontext_est=1258 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/gemma-4-31b-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=64.68% | missing sections: title, description, keywords | missing terms: wooden, style, extends, over, Its
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=773 | text_est=439 | nontext_est=334 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16753 | text_est=439 | nontext_est=16314 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.98% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16753 | text_est=439 | nontext_est=16314 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.
