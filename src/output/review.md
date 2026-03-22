# Automated Review Digest

_Generated on 2026-03-22 02:17:59 GMT_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

**Review artifacts:**

- Standalone output gallery: [model_gallery.md](model_gallery.md)
- Canonical run log: [check_models.log](check_models.log)

## Maintainer Queue

### `mlx`

- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, utility_delta_neutral, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/X-Reasoner-7B-8bit`: `cutoff` | token_cap, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-vlm`

- `mlx-community/Qwen3.5-27B-4bit`: `harness` | harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `mlx-community/Qwen3.5-27B-mxfp8`: `harness` | harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `mlx-community/Qwen3.5-35B-A3B-6bit`: `harness` | harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `mlx-community/Qwen3.5-35B-A3B-bf16`: `harness` | harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `prince-canuma/Florence-2-large-ft`: `harness` | harness:stop_token, missing_sections, low_hint_overlap, degeneration | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: `harness` | harness:encoding, trusted_overlap | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `microsoft/Phi-3.5-vision-instruct`: `harness` | harness:stop_token, trusted_overlap, metadata_terms | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: `harness` | harness:stop_token, utility_delta_positive, metadata_terms | Inspect prompt-template, stop-token, and decode post-processing behavior.

### `model`

- `microsoft/Florence-2-large-ft`: `model_shortcoming` | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/InternVL3-8B-bf16`: `model_shortcoming` | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Molmo-7B-D-0924-bf16`: `model_shortcoming` | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/deepseek-vl2-8bit`: `model_shortcoming` | contract, low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/LFM2-VL-1.6B-8bit`: `model_shortcoming` | metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/FastVLM-0.5B-bf16`: `model_shortcoming` | contract, utility:D, low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/nanoLLaVA-1.5-4bit`: `model_shortcoming` | metadata_borrowing, utility:D, trusted_overlap, metadata_terms, context_echo | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: `model_shortcoming` | contract, utility:F, low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: `clean` | utility_delta_positive | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: `model_shortcoming` | metadata_borrowing, trusted_overlap, metadata_terms | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: `model_shortcoming` | metadata_borrowing, contract, low_hint_overlap, metadata_terms | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/LFM2.5-VL-1.6B-bf16`: `cutoff` | token_cap, abrupt_tail, trusted_overlap, metadata_terms, context_echo | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/InternVL3-14B-8bit`: `clean` | low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/gemma-3-27b-it-qat-4bit`: `clean` | trusted_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: `cutoff` | token_cap, missing_sections, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/pixtral-12b-8bit`: `clean` | trusted_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/gemma-3n-E2B-4bit`: `cutoff` | token_cap, missing_sections, low_hint_overlap, metadata_terms | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `qnguyen3/nanoLLaVA`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/SmolVLM-Instruct-bf16`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: `clean` | trusted_overlap | Treat as a model-quality limitation for this prompt and image.
- `HuggingFaceTB/SmolVLM-Instruct`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: `clean` | trusted_overlap | Treat as a model-quality limitation for this prompt and image.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/llava-v1.6-mistral-7b-8bit`: `model_shortcoming` | metadata_borrowing, contract, trusted_overlap, metadata_terms | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/gemma-3-27b-it-qat-8bit`: `clean` | utility_delta_negative | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/gemma-3n-E4B-it-bf16`: `model_shortcoming` | contract, utility_delta_positive | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: `cutoff` | token_cap, missing_sections, low_hint_overlap, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/paligemma2-3b-pt-896-4bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, unfinished_section, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Idefics3-8B-Llama3-bf16`: `cutoff` | token_cap, missing_sections, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/GLM-4.6V-Flash-6bit`: `cutoff` | token_cap, missing_sections, trusted_overlap, metadata_terms, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: `cutoff` | token_cap, missing_sections, utility_delta_positive, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `Qwen/Qwen3-VL-2B-Instruct`: `cutoff` | token_cap, abrupt_tail, trusted_overlap, context_echo | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: `model_shortcoming` | metadata_borrowing, contract, utility:D, trusted_overlap, metadata_terms | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/pixtral-12b-bf16`: `cutoff` | token_cap, missing_sections, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Molmo-7B-D-0924-8bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, trusted_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/GLM-4.6V-nvfp4`: `cutoff` | token_cap, missing_sections, trusted_overlap, metadata_terms, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, trusted_overlap, context_echo | Inspect token cap and stop behavior before treating this as a model-quality failure.

### `model-config`

- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: `harness` | harness:prompt_template, low_hint_overlap | Inspect model repo config, chat template, and EOS settings.
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: `harness` | harness:prompt_template, low_hint_overlap | Inspect model repo config, chat template, and EOS settings.

## User Buckets

### `recommended`

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: `clean` | improves trusted hints | utility_delta_positive
- `mlx-community/gemma-3-27b-it-qat-4bit`: `clean` | preserves trusted hints | trusted_overlap
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: `clean` | preserves trusted hints | trusted_overlap

### `caveat`

- `mlx-community/InternVL3-14B-8bit`: `clean` | ignores trusted hints | low_hint_overlap
- `mlx-community/pixtral-12b-8bit`: `clean` | preserves trusted hints | trusted_overlap
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: `clean` | preserves trusted hints | trusted_overlap

### `avoid`

- `microsoft/Florence-2-large-ft`: `model_shortcoming` | preserves trusted hints | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- `mlx-community/InternVL3-8B-bf16`: `model_shortcoming` | preserves trusted hints | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- `mlx-community/Molmo-7B-D-0924-bf16`: `model_shortcoming` | preserves trusted hints | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- `mlx-community/Qwen3.5-27B-4bit`: `harness` | preserves trusted hints | harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- `mlx-community/Qwen3.5-27B-mxfp8`: `harness` | preserves trusted hints | harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- `mlx-community/Qwen3.5-35B-A3B-6bit`: `harness` | preserves trusted hints | harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- `mlx-community/Qwen3.5-35B-A3B-bf16`: `harness` | preserves trusted hints | harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- `mlx-community/deepseek-vl2-8bit`: `model_shortcoming` | ignores trusted hints | contract, low_hint_overlap
- `mlx-community/LFM2-VL-1.6B-8bit`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo
- `mlx-community/FastVLM-0.5B-bf16`: `model_shortcoming` | ignores trusted hints | contract, utility:D, low_hint_overlap
- `mlx-community/nanoLLaVA-1.5-4bit`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, utility:D, trusted_overlap, metadata_terms, context_echo
- `prince-canuma/Florence-2-large-ft`: `harness` | ignores trusted hints | harness:stop_token, missing_sections, low_hint_overlap, degeneration
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: `model_shortcoming` | ignores trusted hints | contract, utility:F, low_hint_overlap
- `mlx-community/Phi-3.5-vision-instruct-bf16`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, trusted_overlap, metadata_terms
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: `model_shortcoming` | ignores trusted hints | metadata_borrowing, contract, low_hint_overlap, metadata_terms
- `mlx-community/LFM2.5-VL-1.6B-bf16`: `cutoff` | preserves trusted hints | token_cap, abrupt_tail, trusted_overlap, metadata_terms, context_echo
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: `harness` | ignores trusted hints | harness:prompt_template, low_hint_overlap
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap
- `mlx-community/gemma-3n-E2B-4bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap, metadata_terms
- `qnguyen3/nanoLLaVA`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/SmolVLM-Instruct-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `HuggingFaceTB/SmolVLM-Instruct`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/llava-v1.6-mistral-7b-8bit`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, contract, trusted_overlap, metadata_terms
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/gemma-3-27b-it-qat-8bit`: `clean` | degrades trusted hints | utility_delta_negative
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: `harness` | preserves trusted hints | harness:encoding, trusted_overlap
- `mlx-community/gemma-3n-E4B-it-bf16`: `model_shortcoming` | improves trusted hints | contract, utility_delta_positive
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap, degeneration
- `microsoft/Phi-3.5-vision-instruct`: `harness` | preserves trusted hints | harness:stop_token, trusted_overlap, metadata_terms
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: `harness` | improves trusted hints | harness:stop_token, utility_delta_positive, metadata_terms
- `mlx-community/paligemma2-3b-pt-896-4bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, unfinished_section, abrupt_tail, low_hint_overlap
- `mlx-community/GLM-4.6V-Flash-mxfp4`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap, reasoning_leak
- `mlx-community/Idefics3-8B-Llama3-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap
- `mlx-community/GLM-4.6V-Flash-6bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, trusted_overlap, metadata_terms, reasoning_leak
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: `cutoff` | improves trusted hints | token_cap, missing_sections, utility_delta_positive, reasoning_leak
- `Qwen/Qwen3-VL-2B-Instruct`: `cutoff` | preserves trusted hints | token_cap, abrupt_tail, trusted_overlap, context_echo
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, contract, utility:D, trusted_overlap, metadata_terms
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: `cutoff` | preserves trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, utility_delta_neutral, reasoning_leak
- `mlx-community/X-Reasoner-7B-8bit`: `cutoff` | ignores trusted hints | token_cap, abrupt_tail, low_hint_overlap
- `mlx-community/pixtral-12b-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap
- `mlx-community/Molmo-7B-D-0924-8bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, trusted_overlap
- `mlx-community/GLM-4.6V-nvfp4`: `cutoff` | preserves trusted hints | token_cap, missing_sections, trusted_overlap, metadata_terms, reasoning_leak
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, repetitive_tail, trusted_overlap, context_echo
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: `harness` | ignores trusted hints | harness:prompt_template, low_hint_overlap

## Model Verdicts

### `microsoft/Florence-2-large-ft`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=35
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model | package=transformers | stage=Model Error | code=TRANSFORMERS_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=37
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=35
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=37
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=transformers | stage=Model Error | code=TRANSFORMERS_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=37
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=transformers | stage=Model Error | code=TRANSFORMERS_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=37
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=transformers | stage=Model Error | code=TRANSFORMERS_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=37
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=mlx-vlm | harness=stop_token | package=transformers | stage=Model Error | code=TRANSFORMERS_DECODE_MODEL
- **Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/deepseek-vl2-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** contract, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model | package=model-config | stage=Processor Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- **Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=784 | text_est=467 | nontext_est=317 | gen=93 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** contract, utility:D, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=533 | text_est=467 | nontext_est=66 | gen=28 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, utility:D, trusted_overlap, metadata_terms, context_echo
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=11 | description sentences=3 | keywords=3
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=529 | text_est=467 | nontext_est=62 | gen=500 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `prince-canuma/Florence-2-large-ft`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, missing_sections, low_hint_overlap, degeneration
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1099 | text_est=467 | nontext_est=632 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** contract, utility:F, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, dusk
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1556 | text_est=467 | nontext_est=1089 | gen=17 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** clean | user=recommended
- **Why:** utility_delta_positive
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3123 | text_est=467 | nontext_est=2656 | gen=142 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1350 | text_est=467 | nontext_est=883 | gen=123 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, contract, low_hint_overlap, metadata_terms
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1556 | text_est=467 | nontext_est=1089 | gen=42 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, abrupt_tail, trusted_overlap, metadata_terms, context_echo
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=15 | description sentences=3 | keywords=165 | keyword duplication=0.77
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=784 | text_est=467 | nontext_est=317 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** harness:prompt_template, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1556 | text_est=467 | nontext_est=1089 | gen=9 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** title words=2
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2312 | text_est=467 | nontext_est=1845 | gen=64 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** clean | user=recommended
- **Why:** trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=800 | text_est=467 | nontext_est=333 | gen=87 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=637 | text_est=467 | nontext_est=170 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** clean | user=caveat
- **Why:** trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** title words=4
- **Utility:** user=caveat | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3314 | text_est=467 | nontext_est=2847 | gen=97 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap, metadata_terms
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=791 | text_est=467 | nontext_est=324 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=529 | text_est=467 | nontext_est=62 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1736 | text_est=467 | nontext_est=1269 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=caveat
- **Why:** trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** description sentences=3
- **Utility:** user=caveat | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3124 | text_est=467 | nontext_est=2657 | gen=128 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1736 | text_est=467 | nontext_est=1269 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** clean | user=recommended
- **Why:** trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3124 | text_est=467 | nontext_est=2657 | gen=130 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1522 | text_est=467 | nontext_est=1055 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, contract, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2732 | text_est=467 | nontext_est=2265 | gen=182 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1522 | text_est=467 | nontext_est=1055 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** clean | user=avoid
- **Why:** utility_delta_negative
- **Trusted hints:** degrades trusted hints
- **Contract:** ok
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=800 | text_est=467 | nontext_est=333 | gen=89 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** harness:encoding, trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: description, keywords | title words=53
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2626 | text_est=467 | nontext_est=2159 | gen=93 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** contract, utility_delta_positive
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=799 | text_est=467 | nontext_est=332 | gen=355 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap, degeneration
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1522 | text_est=467 | nontext_est=1055 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=45
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1350 | text_est=467 | nontext_est=883 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, utility_delta_positive, metadata_terms
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** keywords=34 | keyword duplication=0.35
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1845 | text_est=467 | nontext_est=1378 | gen=464 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, unfinished_section, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4628 | text_est=467 | nontext_est=4161 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, reasoning_leak
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6643 | text_est=467 | nontext_est=6176 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2811 | text_est=467 | nontext_est=2344 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, trusted_overlap, metadata_terms, reasoning_leak
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6643 | text_est=467 | nontext_est=6176 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, utility_delta_positive, reasoning_leak
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, keywords
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3405 | text_est=467 | nontext_est=2938 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, abrupt_tail, trusted_overlap, context_echo
- **Trusted hints:** preserves trusted hints
- **Contract:** title words=50 | keywords=173 | keyword duplication=0.84
- **Utility:** user=avoid | preserves trusted hints | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16746 | text_est=467 | nontext_est=16279 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, contract, utility:D, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=501 | text_est=467 | nontext_est=34 | gen=100 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, utility_delta_neutral, reasoning_leak
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16748 | text_est=467 | nontext_est=16281 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** title words=4 | keywords=100 | keyword duplication=0.57
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16757 | text_est=467 | nontext_est=16290 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3314 | text_est=467 | nontext_est=2847 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1708 | text_est=467 | nontext_est=1241 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, trusted_overlap, metadata_terms, reasoning_leak
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6643 | text_est=467 | nontext_est=6176 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, trusted_overlap, context_echo
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=500 | text_est=467 | nontext_est=33 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** harness | user=avoid
- **Why:** harness:prompt_template, low_hint_overlap
- **Trusted hints:** ignores trusted hints
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=0 | text_est=467 | nontext_est=0 | gen=0 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.
