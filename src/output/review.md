# Automated Review Digest

_Generated on 2026-03-28 11:07:10 GMT_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [check_models.log](check_models.log)

## Maintainer Queue

### `mlx`

- `Qwen/Qwen3-VL-2B-Instruct`: `cutoff` | token_cap, repetitive_tail, abrupt_tail, trusted_overlap, context_echo | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap, reasoning_leak, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/X-Reasoner-7B-8bit`: `cutoff` | token_cap, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-35B-A3B-bf16`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-vlm`

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: `harness` | harness:encoding, trusted_overlap, generic | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `microsoft/Phi-3.5-vision-instruct`: `harness` | harness:stop_token, trusted_overlap | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: `harness` | harness:stop_token, low_hint_overlap | Inspect prompt-template, stop-token, and decode post-processing behavior.

### `model`

- `mlx-community/deepseek-vl2-8bit`: `model_shortcoming` | contract, low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: `model_shortcoming` | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: `model_shortcoming` | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: `model_shortcoming` | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/paligemma2-3b-pt-896-4bit`: `model_shortcoming` | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/LFM2-VL-1.6B-8bit`: `model_shortcoming` | metadata_borrowing, contract, utility:D, trusted_overlap, metadata_terms | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/nanoLLaVA-1.5-4bit`: `model_shortcoming` | contract, utility:F, trusted_overlap, context_echo | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/LFM2.5-VL-1.6B-bf16`: `model_shortcoming` | metadata_borrowing, utility:D, trusted_overlap, metadata_terms, context_echo | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/FastVLM-0.5B-bf16`: `model_shortcoming` | hallucination, contract, low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: `clean` | utility_delta_positive | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: `clean` | trusted_overlap, context_echo | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: `clean` | trusted_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/InternVL3-14B-8bit`: `clean` | low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/gemma-3-27b-it-qat-4bit`: `clean` | trusted_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/gemma-3n-E2B-4bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, metadata_terms | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: `clean` | trusted_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: `cutoff` | token_cap, missing_sections, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `HuggingFaceTB/SmolVLM-Instruct`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `qnguyen3/nanoLLaVA`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/SmolVLM-Instruct-bf16`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/pixtral-12b-8bit`: `clean` | trusted_overlap, context_echo | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/gemma-3-27b-it-qat-8bit`: `clean` | trusted_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/gemma-3n-E4B-it-bf16`: `model_shortcoming` | metadata_borrowing, utility_delta_positive, metadata_terms | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: `cutoff` | token_cap, missing_sections, trusted_overlap, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: `cutoff` | token_cap, missing_sections, abrupt_tail, utility_delta_positive, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/GLM-4.6V-Flash-6bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, utility_delta_positive, reasoning_leak, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Idefics3-8B-Llama3-bf16`: `cutoff` | token_cap, missing_sections, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/InternVL3-8B-bf16`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, trusted_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/GLM-4.6V-nvfp4`: `cutoff` | token_cap, missing_sections, utility_delta_positive, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Molmo-7B-D-0924-8bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms, fabrication | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/pixtral-12b-bf16`: `cutoff` | token_cap, missing_sections, low_hint_overlap, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: `model_shortcoming` | contract, trusted_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Qwen3.5-35B-A3B-6bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal, fabrication | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Molmo-7B-D-0924-bf16`: `cutoff` | token_cap, missing_sections, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-27B-4bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms, refusal, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-27B-mxfp8`: `context_budget` | nontext_prompt_burden, utility_delta_positive, refusal | Reduce prompt/image burden or inspect long-context handling before judging quality.

### `model-config`

- `mlx-community/MolmoPoint-8B-fp16`: `harness` | processor_error, model_config_processor_load_processor | Inspect model repo config, chat template, and EOS settings.
- `mlx-community/llava-v1.6-mistral-7b-8bit`: `harness` | harness:prompt_template, low_hint_overlap | Inspect model repo config, chat template, and EOS settings.

### `transformers`

- `microsoft/Florence-2-large-ft`: `harness` | model_error, transformers_model_load_model | Inspect upstream template/tokenizer/config compatibility.
- `prince-canuma/Florence-2-large-ft`: `harness` | model_error, transformers_model_load_model | Inspect upstream template/tokenizer/config compatibility.

## User Buckets

### `recommended`

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: `clean` | preserves trusted hints | trusted_overlap
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: `clean` | preserves trusted hints | trusted_overlap
- `mlx-community/gemma-3-27b-it-qat-8bit`: `clean` | preserves trusted hints | trusted_overlap

### `caveat`

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: `clean` | improves trusted hints | utility_delta_positive
- `mlx-community/Phi-3.5-vision-instruct-bf16`: `clean` | preserves trusted hints | trusted_overlap, context_echo
- `mlx-community/InternVL3-14B-8bit`: `clean` | ignores trusted hints | low_hint_overlap
- `mlx-community/gemma-3-27b-it-qat-4bit`: `clean` | preserves trusted hints | trusted_overlap
- `mlx-community/pixtral-12b-8bit`: `clean` | preserves trusted hints | trusted_overlap, context_echo
- `mlx-community/Qwen3.5-27B-mxfp8`: `context_budget` | improves trusted hints | nontext_prompt_burden, utility_delta_positive, refusal

### `avoid`

- `microsoft/Florence-2-large-ft`: `harness` | preserves trusted hints | model_error, transformers_model_load_model
- `mlx-community/MolmoPoint-8B-fp16`: `harness` | preserves trusted hints | processor_error, model_config_processor_load_processor
- `mlx-community/deepseek-vl2-8bit`: `model_shortcoming` | ignores trusted hints | contract, low_hint_overlap
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: `model_shortcoming` | preserves trusted hints | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: `model_shortcoming` | preserves trusted hints | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: `model_shortcoming` | preserves trusted hints | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- `mlx-community/paligemma2-3b-pt-896-4bit`: `model_shortcoming` | preserves trusted hints | instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- `prince-canuma/Florence-2-large-ft`: `harness` | preserves trusted hints | model_error, transformers_model_load_model
- `mlx-community/LFM2-VL-1.6B-8bit`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, contract, utility:D, trusted_overlap, metadata_terms
- `mlx-community/nanoLLaVA-1.5-4bit`: `model_shortcoming` | preserves trusted hints | contract, utility:F, trusted_overlap, context_echo
- `mlx-community/LFM2.5-VL-1.6B-bf16`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, utility:D, trusted_overlap, metadata_terms, context_echo
- `mlx-community/FastVLM-0.5B-bf16`: `model_shortcoming` | ignores trusted hints | hallucination, contract, low_hint_overlap
- `mlx-community/llava-v1.6-mistral-7b-8bit`: `harness` | ignores trusted hints | harness:prompt_template, low_hint_overlap
- `mlx-community/gemma-3n-E2B-4bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, metadata_terms
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap
- `HuggingFaceTB/SmolVLM-Instruct`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `qnguyen3/nanoLLaVA`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/SmolVLM-Instruct-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap, degeneration
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: `harness` | preserves trusted hints | harness:encoding, trusted_overlap, generic
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap
- `mlx-community/gemma-3n-E4B-it-bf16`: `model_shortcoming` | improves trusted hints | metadata_borrowing, utility_delta_positive, metadata_terms
- `microsoft/Phi-3.5-vision-instruct`: `harness` | preserves trusted hints | harness:stop_token, trusted_overlap
- `mlx-community/GLM-4.6V-Flash-mxfp4`: `cutoff` | preserves trusted hints | token_cap, missing_sections, trusted_overlap, reasoning_leak
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: `cutoff` | improves trusted hints | token_cap, missing_sections, abrupt_tail, utility_delta_positive, reasoning_leak
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap
- `mlx-community/GLM-4.6V-Flash-6bit`: `cutoff` | improves trusted hints | token_cap, missing_sections, abrupt_tail, utility_delta_positive, reasoning_leak, degeneration
- `mlx-community/Idefics3-8B-Llama3-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap
- `Qwen/Qwen3-VL-2B-Instruct`: `cutoff` | preserves trusted hints | token_cap, repetitive_tail, abrupt_tail, trusted_overlap, context_echo
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap, reasoning_leak, degeneration
- `mlx-community/InternVL3-8B-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, degeneration
- `mlx-community/X-Reasoner-7B-8bit`: `cutoff` | ignores trusted hints | token_cap, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, repetitive_tail, trusted_overlap
- `mlx-community/GLM-4.6V-nvfp4`: `cutoff` | improves trusted hints | token_cap, missing_sections, utility_delta_positive, reasoning_leak
- `mlx-community/Molmo-7B-D-0924-8bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms, fabrication
- `mlx-community/pixtral-12b-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap, degeneration
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: `model_shortcoming` | preserves trusted hints | contract, trusted_overlap
- `mlx-community/Qwen3.5-35B-A3B-6bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal, fabrication
- `mlx-community/Qwen3.5-35B-A3B-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap
- `mlx-community/Molmo-7B-D-0924-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: `harness` | ignores trusted hints | harness:stop_token, low_hint_overlap
- `mlx-community/Qwen3.5-27B-4bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms, refusal, degeneration

## Model Verdicts

### `microsoft/Florence-2-large-ft`

- **Verdict:** harness | user=avoid
- **Why:** model_error, transformers_model_load_model
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=transformers | package=transformers | stage=Model Error | code=TRANSFORMERS_MODEL_LOAD_MODEL
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect upstream template/tokenizer/config compatibility.

### `mlx-community/MolmoPoint-8B-fp16`

- **Verdict:** harness | user=avoid
- **Why:** processor_error, model_config_processor_load_processor
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=model-config | package=model-config | stage=Processor Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/deepseek-vl2-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** contract, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model | package=model-config | stage=Processor Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- **Token accounting:** prompt=n/a | text_est=437 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=35
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model | package=transformers | stage=API Mismatch | code=TRANSFORMERS_DECODE_API_MISMATCH
- **Token accounting:** prompt=n/a | text_est=437 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=35
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model | package=transformers | stage=API Mismatch | code=TRANSFORMERS_DECODE_API_MISMATCH
- **Token accounting:** prompt=n/a | text_est=437 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=35
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model | package=transformers | stage=API Mismatch | code=TRANSFORMERS_DECODE_API_MISMATCH
- **Token accounting:** prompt=n/a | text_est=437 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=29 | description sentences=3 | keywords=35
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model | package=transformers | stage=API Mismatch | code=TRANSFORMERS_DECODE_API_MISMATCH
- **Token accounting:** prompt=n/a | text_est=437 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `prince-canuma/Florence-2-large-ft`

- **Verdict:** harness | user=avoid
- **Why:** model_error, transformers_model_load_model
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=transformers | package=transformers | stage=Model Error | code=TRANSFORMERS_MODEL_LOAD_MODEL
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect upstream template/tokenizer/config compatibility.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, contract, utility:D, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=760 | text_est=437 | nontext_est=323 | gen=85 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** contract, utility:F, trusted_overlap, context_echo
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: keywords | title words=11 | description sentences=3
- **Utility:** user=avoid | preserves trusted hints | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=506 | text_est=437 | nontext_est=69 | gen=84 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, utility:D, trusted_overlap, metadata_terms, context_echo
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=23 | description sentences=3
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=570 | text_est=437 | nontext_est=133 | gen=166 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** hallucination, contract, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, former, warehouse, London, Canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=510 | text_est=437 | nontext_est=73 | gen=118 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** clean | user=caveat
- **Why:** utility_delta_positive
- **Trusted hints:** improves trusted hints
- **Contract:** description sentences=3
- **Utility:** user=caveat | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3115 | text_est=437 | nontext_est=2678 | gen=143 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** harness | user=avoid
- **Why:** harness:prompt_template, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=2726 | text_est=437 | nontext_est=2289 | gen=8 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** clean | user=caveat
- **Why:** trusted_overlap, context_echo
- **Trusted hints:** preserves trusted hints
- **Contract:** description sentences=3
- **Utility:** user=caveat | preserves trusted hints | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1326 | text_est=437 | nontext_est=889 | gen=141 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=recommended
- **Why:** trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3116 | text_est=437 | nontext_est=2679 | gen=97 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** title words=2
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2289 | text_est=437 | nontext_est=1852 | gen=71 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** clean | user=caveat
- **Why:** trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** keywords=19
- **Utility:** user=caveat | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=777 | text_est=437 | nontext_est=340 | gen=94 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, metadata_terms
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=768 | text_est=437 | nontext_est=331 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** clean | user=recommended
- **Why:** trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3116 | text_est=437 | nontext_est=2679 | gen=113 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=614 | text_est=437 | nontext_est=177 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1714 | text_est=437 | nontext_est=1277 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=506 | text_est=437 | nontext_est=69 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1714 | text_est=437 | nontext_est=1277 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap, degeneration
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1497 | text_est=437 | nontext_est=1060 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** clean | user=caveat
- **Why:** trusted_overlap, context_echo
- **Trusted hints:** preserves trusted hints
- **Contract:** description sentences=3
- **Utility:** user=caveat | preserves trusted hints | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3309 | text_est=437 | nontext_est=2872 | gen=122 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1497 | text_est=437 | nontext_est=1060 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** clean | user=recommended
- **Why:** trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=777 | text_est=437 | nontext_est=340 | gen=92 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** harness:encoding, trusted_overlap, generic
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: description, keywords | title words=65
- **Utility:** user=avoid | preserves trusted hints | generic
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2618 | text_est=437 | nontext_est=2181 | gen=104 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1497 | text_est=437 | nontext_est=1060 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, utility_delta_positive, metadata_terms
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** description sentences=6 | keywords=30
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=776 | text_est=437 | nontext_est=339 | gen=347 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** description sentences=3 | keywords=42
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1326 | text_est=437 | nontext_est=889 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, trusted_overlap, reasoning_leak
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6524 | text_est=437 | nontext_est=6087 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, utility_delta_positive, reasoning_leak
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, keywords | description sentences=7
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3400 | text_est=437 | nontext_est=2963 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1822 | text_est=437 | nontext_est=1385 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, utility_delta_positive, reasoning_leak, degeneration
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title | keywords=47 | keyword duplication=0.51
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6524 | text_est=437 | nontext_est=6087 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2789 | text_est=437 | nontext_est=2352 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, repetitive_tail, abrupt_tail, trusted_overlap, context_echo
- **Trusted hints:** preserves trusted hints
- **Contract:** title words=23 | description sentences=3 | keywords=152 | keyword duplication=0.86
- **Utility:** user=avoid | preserves trusted hints | context echo
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16723 | text_est=437 | nontext_est=16286 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap, reasoning_leak, degeneration
- **Trusted hints:** ignores trusted hints | missing terms: brick, former, warehouse, London, Canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16725 | text_est=437 | nontext_est=16288 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, degeneration
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2289 | text_est=437 | nontext_est=1852 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** title words=3 | keywords=94 | keyword duplication=0.68
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16734 | text_est=437 | nontext_est=16297 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=477 | text_est=437 | nontext_est=40 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, utility_delta_positive, reasoning_leak
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6524 | text_est=437 | nontext_est=6087 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1693 | text_est=437 | nontext_est=1256 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap, degeneration
- **Trusted hints:** ignores trusted hints | missing terms: large, former, warehouse, London, Canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3309 | text_est=437 | nontext_est=2872 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** contract, trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=478 | text_est=437 | nontext_est=41 | gen=159 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16749 | text_est=437 | nontext_est=16312 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: brick, former, warehouse, London, Canal
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16749 | text_est=437 | nontext_est=16312 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1693 | text_est=437 | nontext_est=1256 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: large, brick, former, warehouse, London
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=16734 | text_est=437 | nontext_est=16297 | gen=4 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms, refusal, degeneration
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16749 | text_est=437 | nontext_est=16312 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** context_budget | user=caveat
- **Why:** nontext_prompt_burden, utility_delta_positive, refusal
- **Trusted hints:** improves trusted hints
- **Contract:** keywords=26
- **Utility:** user=caveat | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16749 | text_est=437 | nontext_est=16312 | gen=500 | max=500 | stop=completed
- **Next action:** Reduce prompt/image burden or inspect long-context handling before judging quality.
