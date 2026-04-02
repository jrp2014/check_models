# Automated Review Digest

_Generated on 2026-04-02 23:22:04 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [check_models.log](check_models.log)

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🏆 A (94/100) | Δ+16 | 184.8 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (90/100) | Δ+13 | 63.2 tps

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) | Δ-71 | 31.4 tps | harness, missing sections, trusted hint degraded
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (10/100) | Δ-68 | 67.5 tps | context ignored, harness, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (17/100) | Δ-61 | 5.8 tps | context ignored, harness
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (20/100) | Δ-58 | 21.1 tps | context ignored, harness, missing sections
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (23/100) | Δ-55 | 202.2 tps | context ignored, hallucination, harness, long context, missing sections

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | improves trusted hints | nontext prompt burden=84% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Handrail, Objects |

### `caveat`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | improves trusted hints | nontext prompt burden=84% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Handrail, Objects |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Handrail | nontext prompt burden=84% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Handrail |
| `mlx-community/InternVL3-14B-8bit` | `clean` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | nontext prompt burden=79% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture |
| `mlx-community/X-Reasoner-7B-8bit` | `context_budget` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | At long prompt length (16824 tokens), output may stop following prompt/image context. \| output/prompt=0.93% \| nontext prompt burden=97% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | At long prompt length (16824 tokens), output may stop following prompt/image context. \| output/prompt=0.11% \| nontext prompt burden=97% \| missing sections: title, description, keywords |

### `avoid`

| Model | Verdict | Hint Handling | Key Evidence |
| ----- | ------- | ------------- | ------------ |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | not evaluated | processor error \| model config processor load processor |
| `mlx-community/LFM2-VL-1.6B-8bit` | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused | missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight \| context echo=98% \| nonvisual metadata reused |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `harness` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=69% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture |
| `mlx-community/nanoLLaVA-1.5-4bit` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=83.89% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight \| keyword duplication=93% |
| `mlx-community/FastVLM-0.5B-bf16` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=83.33% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `model_shortcoming` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | nontext prompt burden=69% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=75.64% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Handrail, Modern Art \| keyword duplication=89% |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `harness` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight | Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=82% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=69% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `clean` | degrades trusted hints | missing terms: 10 Best (structured), Modern Art, Objects, Royston, Statue |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=70.62% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=27.65% \| nontext prompt burden=72% \| missing sections: title, description, keywords |
| `mlx-community/pixtral-12b-8bit` | `clean` | degrades trusted hints | nontext prompt burden=85% \| missing terms: 10 Best (structured), Abstract Art, Modern Art, Objects, Royston |
| `mlx-community/gemma-3n-E2B-4bit` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture \| nonvisual metadata reused | hit token cap (500) \| output/prompt=58.55% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=31.57% \| nontext prompt burden=68% \| missing sections: title, description, keywords |
| `qnguyen3/nanoLLaVA` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=83.89% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=27.65% \| nontext prompt burden=72% \| missing sections: title, description, keywords |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=31.57% \| nontext prompt burden=68% \| missing sections: title, description, keywords |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `clean` | degrades trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Modern Art, Objects |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | degrades trusted hints | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 46 occurrences). \| nontext prompt burden=82% \| missing sections: description, keywords \| missing terms: 10 Best (structured), Abstract Art, Handrail, Objects, Royston |
| `mlx-community/gemma-3n-E4B-it-bf16` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | missing terms: 10 Best (structured), Bronze Sculpture, Handrail, Modern Art, Objects \| nonvisual metadata reused |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=31.57% \| nontext prompt burden=68% \| missing sections: title, description, keywords |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=35.04% \| nontext prompt burden=65% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail |
| `microsoft/Phi-3.5-vision-instruct` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=35.04% \| nontext prompt burden=65% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=10.66% \| nontext prompt burden=89% \| missing sections: title, description, keywords |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `harness` | improves trusted hints \| nonvisual metadata reused | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, keywords |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | improves trusted hints | hit token cap (500) \| output/prompt=14.33% \| nontext prompt burden=86% \| missing sections: title |
| `mlx-community/GLM-4.6V-Flash-6bit` | `cutoff` | degrades trusted hints | hit token cap (500) \| output/prompt=7.45% \| nontext prompt burden=93% \| missing sections: title, description, keywords |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=17.39% \| nontext prompt burden=83% \| missing sections: title, description, keywords |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff` | degrades trusted hints | hit token cap (500) \| output/prompt=26.03% \| nontext prompt burden=74% \| missing sections: title, description, keywords |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | At long prompt length (16813 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail |
| `mlx-community/InternVL3-8B-bf16` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=21.02% \| nontext prompt burden=79% \| missing sections: title, description, keywords |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `model_shortcoming` | preserves trusted hints \| nonvisual metadata reused | missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Daylight, Modern Art, Objects \| nonvisual metadata reused |
| `mlx-community/Molmo-7B-D-0924-8bit` | `model_shortcoming` | improves trusted hints \| nonvisual metadata reused | nontext prompt burden=72% \| missing terms: 10 Best (structured), Abstract Art, Modern Art, Objects, Royston \| keywords=9 \| nonvisual metadata reused |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `cutoff` | degrades trusted hints | hit token cap (500) \| output/prompt=88.65% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail |
| `mlx-community/GLM-4.6V-nvfp4` | `harness` | improves trusted hints \| nonvisual metadata reused | Special control token &lt;/think&gt; appeared in generated text. \| nontext prompt burden=93% \| missing sections: title \| missing terms: 10 Best (structured), Objects, Royston, 'Maquette, Spirit |
| `mlx-community/pixtral-12b-bf16` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight | hit token cap (500) \| output/prompt=14.71% \| nontext prompt burden=85% \| missing sections: title, description, keywords |
| `mlx-community/Molmo-7B-D-0924-bf16` | `cutoff` | ignores trusted hints \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | hit token cap (500) \| output/prompt=28.04% \| nontext prompt burden=72% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: keywords |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-27B-4bit` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: title, description, keywords |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | preserves trusted hints \| nonvisual metadata reused | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: title, description, keywords |

## Maintainer Queue

Owner-grouped escalations with compact evidence and row-specific next actions.

### `mlx`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `Qwen/Qwen3-VL-2B-Instruct` | `cutoff` | At long prompt length (16813 tokens), output became repetitive. \| hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.97%. |
| `mlx-community/X-Reasoner-7B-8bit` | `context_budget` | At long prompt length (16824 tokens), output may stop following prompt/image context. \| output/prompt=0.93% \| nontext prompt burden=97% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load. |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `context_budget` | At long prompt length (16824 tokens), output may stop following prompt/image context. \| output/prompt=0.11% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load. |

### `mlx-vlm`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | `harness` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 46 occurrences). \| nontext prompt burden=82% \| missing sections: description, keywords \| missing terms: 10 Best (structured), Abstract Art, Handrail, Objects, Royston | Inspect decode cleanup; tokenizer markers are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| hit token cap (500) \| nontext prompt burden=93% \| missing sections: title, keywords | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |
| `mlx-community/GLM-4.6V-nvfp4` | `harness` | Special control token &lt;/think&gt; appeared in generated text. \| nontext prompt burden=93% \| missing sections: title \| missing terms: 10 Best (structured), Objects, Royston, 'Maquette, Spirit | Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text. |

### `model`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/LFM2-VL-1.6B-8bit` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight \| context echo=98% \| nonvisual metadata reused | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/nanoLLaVA-1.5-4bit` | `cutoff` | hit token cap (500) \| output/prompt=83.89% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight \| keyword duplication=93% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=83.89%. |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | `clean` | nontext prompt burden=84% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Handrail, Objects | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/FastVLM-0.5B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=83.33% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | `model_shortcoming` | nontext prompt burden=69% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=75.64% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Handrail, Modern Art \| keyword duplication=89% | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=75.64%. |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | `clean` | nontext prompt burden=84% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Handrail, Objects | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | `clean` | nontext prompt burden=84% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Handrail | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/InternVL3-14B-8bit` | `clean` | nontext prompt burden=79% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-3-27b-it-qat-4bit` | `clean` | missing terms: 10 Best (structured), Modern Art, Objects, Royston, Statue | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | `cutoff` | hit token cap (500) \| output/prompt=70.62% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/SmolVLM-Instruct-bf16` | `cutoff` | hit token cap (500) \| output/prompt=27.65% \| nontext prompt burden=72% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/pixtral-12b-8bit` | `clean` | nontext prompt burden=85% \| missing terms: 10 Best (structured), Abstract Art, Modern Art, Objects, Royston | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-3n-E2B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=58.55% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` | `cutoff` | hit token cap (500) \| output/prompt=31.57% \| nontext prompt burden=68% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `qnguyen3/nanoLLaVA` | `cutoff` | hit token cap (500) \| output/prompt=83.89% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff` | hit token cap (500) \| output/prompt=27.65% \| nontext prompt burden=72% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | `cutoff` | hit token cap (500) \| output/prompt=31.57% \| nontext prompt burden=68% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/gemma-3-27b-it-qat-8bit` | `clean` | missing terms: 10 Best (structured), Abstract Art, Blue sky, Modern Art, Objects | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/gemma-3n-E4B-it-bf16` | `model_shortcoming` | missing terms: 10 Best (structured), Bronze Sculpture, Handrail, Modern Art, Objects \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` | `cutoff` | hit token cap (500) \| output/prompt=31.57% \| nontext prompt burden=68% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Phi-3.5-vision-instruct-bf16` | `cutoff` | hit token cap (500) \| output/prompt=35.04% \| nontext prompt burden=65% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=35.04%. |
| `microsoft/Phi-3.5-vision-instruct` | `cutoff` | hit token cap (500) \| output/prompt=35.04% \| nontext prompt burden=65% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=35.04%. |
| `mlx-community/paligemma2-3b-pt-896-4bit` | `cutoff` | hit token cap (500) \| output/prompt=10.66% \| nontext prompt burden=89% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | `cutoff` | hit token cap (500) \| output/prompt=14.33% \| nontext prompt burden=86% \| missing sections: title | Raise the token cap or trim prompt burden first; generation hit the limit while title remained incomplete. |
| `mlx-community/GLM-4.6V-Flash-6bit` | `cutoff` | hit token cap (500) \| output/prompt=7.45% \| nontext prompt burden=93% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Idefics3-8B-Llama3-bf16` | `cutoff` | hit token cap (500) \| output/prompt=17.39% \| nontext prompt burden=83% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `cutoff` | hit token cap (500) \| output/prompt=26.03% \| nontext prompt burden=74% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | `cutoff` | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail | Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.97%. |
| `mlx-community/InternVL3-8B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=21.02% \| nontext prompt burden=79% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | `model_shortcoming` | missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Daylight, Modern Art, Objects \| nonvisual metadata reused | Treat as a model limitation for this prompt; the requested output contract is not being met. |
| `mlx-community/Molmo-7B-D-0924-8bit` | `model_shortcoming` | nontext prompt burden=72% \| missing terms: 10 Best (structured), Abstract Art, Modern Art, Objects, Royston \| keywords=9 \| nonvisual metadata reused | Treat as a model limitation for this prompt; trusted hint coverage is still weak. |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | `cutoff` | hit token cap (500) \| output/prompt=88.65% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/pixtral-12b-bf16` | `cutoff` | hit token cap (500) \| output/prompt=14.71% \| nontext prompt burden=85% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Molmo-7B-D-0924-bf16` | `cutoff` | hit token cap (500) \| output/prompt=28.04% \| nontext prompt burden=72% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: keywords | Raise the token cap or trim prompt burden first; generation hit the limit while keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `cutoff` | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | `cutoff` | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-27B-4bit` | `cutoff` | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |
| `mlx-community/Qwen3.5-27B-mxfp8` | `cutoff` | hit token cap (500) \| output/prompt=2.97% \| nontext prompt burden=97% \| missing sections: title, description, keywords | Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete. |

### `model-config`

| Model | Verdict | Evidence | Next Action |
| ----- | ------- | -------- | ----------- |
| `mlx-community/MolmoPoint-8B-fp16` | `harness` | processor error \| model config processor load processor | Inspect model repo config, chat template, and EOS settings. |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | `harness` | Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=69% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | Check chat-template and EOS defaults first; the output shape is not matching the requested contract. |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | `harness` | Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=82% \| missing sections: title, description, keywords \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight | Check chat-template and EOS defaults first; the output shape is not matching the requested contract. |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | `harness` | Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues. \| nontext prompt burden=69% \| missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | Inspect model repo config, chat template, and EOS settings. |

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
- **Why:** missing sections: title, description, keywords | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight | context echo=98% | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=851 | text_est=501 | nontext_est=350 | gen=89 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=69% | missing sections: title, description, keywords | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1618 | text_est=501 | nontext_est=1117 | gen=11 | max=500 | stop=completed
- **Next action:** Check chat-template and EOS defaults first; the output shape is not matching the requested contract.

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=83.89% | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight | keyword duplication=93%
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=183 | keyword duplication=0.93
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=596 | text_est=501 | nontext_est=95 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=83.89%.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=84% | missing terms: 10 Best (structured), Abstract Art, Blue sky, Handrail, Objects
- **Trusted hints:** improves trusted hints
- **Contract:** title words=4
- **Utility:** user=caveat | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3205 | text_est=501 | nontext_est=2704 | gen=97 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=83.33% | missing sections: title, description, keywords | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=600 | text_est=501 | nontext_est=99 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=69% | missing sections: title, description, keywords | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1618 | text_est=501 | nontext_est=1117 | gen=20 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=75.64% | missing terms: 10 Best (structured), Abstract Art, Blue sky, Handrail, Modern Art | keyword duplication=89%
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=23 | keywords=106 | keyword duplication=0.89
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=661 | text_est=501 | nontext_est=160 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=75.64%.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=82% | missing sections: title, description, keywords | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=2830 | text_est=501 | nontext_est=2329 | gen=10 | max=500 | stop=completed
- **Next action:** Check chat-template and EOS defaults first; the output shape is not matching the requested contract.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues. | nontext prompt burden=69% | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1618 | text_est=501 | nontext_est=1117 | gen=9 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** clean | user=recommended
- **Why:** nontext prompt burden=84% | missing terms: 10 Best (structured), Abstract Art, Blue sky, Handrail, Objects
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3206 | text_est=501 | nontext_est=2705 | gen=97 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=84% | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Handrail
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Handrail
- **Contract:** ok
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3206 | text_est=501 | nontext_est=2705 | gen=100 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** nontext prompt burden=79% | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** title words=2
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2379 | text_est=501 | nontext_est=1878 | gen=71 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** clean | user=avoid
- **Why:** missing terms: 10 Best (structured), Modern Art, Objects, Royston, Statue
- **Trusted hints:** degrades trusted hints
- **Contract:** ok
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=863 | text_est=501 | nontext_est=362 | gen=87 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=70.62% | missing sections: title, description, keywords | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=708 | text_est=501 | nontext_est=207 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=27.65% | nontext prompt burden=72% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1808 | text_est=501 | nontext_est=1307 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** clean | user=avoid
- **Why:** nontext prompt burden=85% | missing terms: 10 Best (structured), Abstract Art, Modern Art, Objects, Royston
- **Trusted hints:** degrades trusted hints
- **Contract:** description sentences=3
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3399 | text_est=501 | nontext_est=2898 | gen=95 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=58.55% | missing sections: title, description, keywords | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=854 | text_est=501 | nontext_est=353 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=31.57% | nontext prompt burden=68% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1584 | text_est=501 | nontext_est=1083 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=83.89% | missing sections: title, description, keywords | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=596 | text_est=501 | nontext_est=95 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=27.65% | nontext prompt burden=72% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1808 | text_est=501 | nontext_est=1307 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=31.57% | nontext prompt burden=68% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1584 | text_est=501 | nontext_est=1083 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** clean | user=avoid
- **Why:** missing terms: 10 Best (structured), Abstract Art, Blue sky, Modern Art, Objects
- **Trusted hints:** degrades trusted hints
- **Contract:** ok
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=863 | text_est=501 | nontext_est=362 | gen=82 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 46 occurrences). | nontext prompt burden=82% | missing sections: description, keywords | missing terms: 10 Best (structured), Abstract Art, Handrail, Objects, Royston
- **Trusted hints:** degrades trusted hints
- **Contract:** missing: description, keywords | title words=48
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2709 | text_est=501 | nontext_est=2208 | gen=75 | max=500 | stop=completed
- **Next action:** Inspect decode cleanup; tokenizer markers are leaking into user-facing text.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing terms: 10 Best (structured), Bronze Sculpture, Handrail, Modern Art, Objects | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=11 | description sentences=9
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=862 | text_est=501 | nontext_est=361 | gen=294 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=31.57% | nontext prompt burden=68% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1584 | text_est=501 | nontext_est=1083 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=35.04% | nontext prompt burden=65% | missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=63 | keyword duplication=0.62
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1427 | text_est=501 | nontext_est=926 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=35.04%.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=35.04% | nontext prompt burden=65% | missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=63 | keyword duplication=0.62
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1427 | text_est=501 | nontext_est=926 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=35.04%.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=10.66% | nontext prompt burden=89% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4690 | text_est=501 | nontext_est=4189 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext prompt burden=93% | missing sections: title, keywords
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6707 | text_est=501 | nontext_est=6206 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=14.33% | nontext prompt burden=86% | missing sections: title
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title | description sentences=3 | keywords=6
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3490 | text_est=501 | nontext_est=2989 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title remained incomplete.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=7.45% | nontext prompt burden=93% | missing sections: title, description, keywords
- **Trusted hints:** degrades trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6707 | text_est=501 | nontext_est=6206 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=17.39% | nontext prompt burden=83% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2876 | text_est=501 | nontext_est=2375 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=26.03% | nontext prompt burden=74% | missing sections: title, description, keywords
- **Trusted hints:** degrades trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1921 | text_est=501 | nontext_est=1420 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** At long prompt length (16813 tokens), output became repetitive. | hit token cap (500) | output/prompt=2.97% | nontext prompt burden=97%
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=2 | keywords=51 | keyword duplication=0.43
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16813 | text_est=501 | nontext_est=16312 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.97%.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** context_budget | user=caveat
- **Why:** At long prompt length (16824 tokens), output may stop following prompt/image context. | output/prompt=0.93% | nontext prompt burden=97% | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** title words=3 | description sentences=3
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16824 | text_est=501 | nontext_est=16323 | gen=157 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.97% | nontext prompt burden=97% | missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=19
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16815 | text_est=501 | nontext_est=16314 | gen=500 | max=500 | stop=completed
- **Next action:** Treat this as cap-limited output first; generation exhausted the token budget with output/prompt=2.97%.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=21.02% | nontext prompt burden=79% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2379 | text_est=501 | nontext_est=1878 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** missing sections: title, description, keywords | missing terms: 10 Best (structured), Abstract Art, Daylight, Modern Art, Objects | nonvisual metadata reused
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=565 | text_est=501 | nontext_est=64 | gen=112 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; the requested output contract is not being met.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** nontext prompt burden=72% | missing terms: 10 Best (structured), Abstract Art, Modern Art, Objects, Royston | keywords=9 | nonvisual metadata reused
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=11 | keywords=9
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1783 | text_est=501 | nontext_est=1282 | gen=294 | max=500 | stop=completed
- **Next action:** Treat as a model limitation for this prompt; trusted hint coverage is still weak.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=88.65% | missing sections: title, description, keywords | missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail
- **Trusted hints:** degrades trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=564 | text_est=501 | nontext_est=63 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** harness | user=avoid
- **Why:** Special control token &lt;/think&gt; appeared in generated text. | nontext prompt burden=93% | missing sections: title | missing terms: 10 Best (structured), Objects, Royston, 'Maquette, Spirit
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title | keywords=54 | keyword duplication=0.35
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6707 | text_est=501 | nontext_est=6206 | gen=438 | max=500 | stop=completed
- **Next action:** Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=14.71% | nontext prompt burden=85% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3399 | text_est=501 | nontext_est=2898 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=28.04% | nontext prompt burden=72% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1783 | text_est=501 | nontext_est=1282 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.97% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.97% | nontext prompt burden=97% | missing sections: keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords | title words=35 | description sentences=3
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while keywords remained incomplete.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.97% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** At long prompt length (16824 tokens), output may stop following prompt/image context. | output/prompt=0.11% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16824 | text_est=501 | nontext_est=16323 | gen=18 | max=500 | stop=completed
- **Next action:** Treat this as a prompt-budget issue first; nontext prompt burden is 97% and the output stays weak under that load.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.97% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.97% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff | user=avoid
- **Why:** hit token cap (500) | output/prompt=2.97% | nontext prompt burden=97% | missing sections: title, description, keywords
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Raise the token cap or trim prompt burden first; generation hit the limit while title, description, keywords remained incomplete.
