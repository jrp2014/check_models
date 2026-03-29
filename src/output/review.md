# Automated Review Digest

_Generated on 2026-03-29 00:31:08 GMT_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [check_models.log](check_models.log)

## Maintainer Queue

### `mlx`

- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap, instruction_markers, metadata_terms, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/X-Reasoner-7B-8bit`: `cutoff` | token_cap, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-9B-MLX-4bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-vlm`

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: `harness` | harness:encoding, trusted_overlap, metadata_terms | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `microsoft/Phi-3.5-vision-instruct`: `harness` | harness:stop_token, abrupt_tail, utility_delta_positive, metadata_terms, degeneration, fabrication | Inspect prompt-template, stop-token, and decode post-processing behavior.

### `model`

- `mlx-community/deepseek-vl2-8bit`: `model_shortcoming` | contract, low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/LFM2-VL-1.6B-8bit`: `model_shortcoming` | metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/nanoLLaVA-1.5-4bit`: `model_shortcoming` | metadata_borrowing, contract, utility:D, trusted_overlap, metadata_terms, fabrication | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/FastVLM-0.5B-bf16`: `model_shortcoming` | contract, low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: `context_budget` | nontext_prompt_burden, utility_delta_positive, metadata_terms | Reduce prompt/image burden or inspect long-context handling before judging quality.
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: `model_shortcoming` | contract, utility:F, low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/LFM2.5-VL-1.6B-bf16`: `cutoff` | token_cap, repetitive_tail, abrupt_tail, trusted_overlap, metadata_terms, context_echo | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: `model_shortcoming` | metadata_borrowing, trusted_overlap, metadata_terms, fabrication | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/llava-v1.6-mistral-7b-8bit`: `model_shortcoming` | metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/gemma-3n-E2B-4bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap, metadata_terms | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/InternVL3-14B-8bit`: `clean` | low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: `cutoff` | token_cap, missing_sections, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `HuggingFaceTB/SmolVLM-Instruct`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/gemma-3-27b-it-qat-4bit`: `clean` | utility_delta_positive | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: `clean` | utility_delta_positive | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/SmolVLM-Instruct-bf16`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: `context_budget` | nontext_prompt_burden, utility_delta_positive, metadata_terms | Reduce prompt/image burden or inspect long-context handling before judging quality.
- `qnguyen3/nanoLLaVA`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/pixtral-12b-8bit`: `context_budget` | nontext_prompt_burden, trusted_overlap, metadata_terms, context_echo | Reduce prompt/image burden or inspect long-context handling before judging quality.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/gemma-3-27b-it-qat-8bit`: `clean` | utility_delta_positive | Treat as a model-quality limitation for this prompt and image.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: `model_shortcoming` | contract, utility:F, low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/gemma-3n-E4B-it-bf16`: `model_shortcoming` | metadata_borrowing, contract, utility_delta_positive, metadata_terms | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: `model_shortcoming` | metadata_borrowing, contract, trusted_overlap, metadata_terms | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/paligemma2-3b-pt-896-4bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap, reasoning_leak, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: `cutoff` | token_cap, missing_sections, abrupt_tail, utility_delta_positive, metadata_terms, reasoning_leak, fabrication | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/InternVL3-8B-bf16`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: `cutoff` | token_cap, missing_sections, utility_delta_positive, metadata_terms | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/GLM-4.6V-Flash-6bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap, reasoning_leak, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Idefics3-8B-Llama3-bf16`: `cutoff` | token_cap, missing_sections, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `Qwen/Qwen3-VL-2B-Instruct`: `cutoff` | token_cap, abrupt_tail, trusted_overlap, metadata_terms, context_echo, fabrication | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Molmo-7B-D-0924-8bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/pixtral-12b-bf16`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/GLM-4.6V-nvfp4`: `cutoff` | token_cap, missing_sections, abrupt_tail, utility_delta_positive, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Molmo-7B-D-0924-bf16`: `cutoff` | token_cap, missing_sections, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-35B-A3B-6bit`: `cutoff` | token_cap, missing_sections, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-35B-A3B-bf16`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms, refusal, fabrication | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-27B-4bit`: `cutoff` | token_cap, missing_sections, trusted_overlap, metadata_terms, refusal, fabrication | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-27B-mxfp8`: `cutoff` | token_cap, missing_sections, trusted_overlap, metadata_terms, refusal | Inspect token cap and stop behavior before treating this as a model-quality failure.

### `model-config`

- `mlx-community/MolmoPoint-8B-fp16`: `harness` | processor_error, model_config_processor_load_processor | Inspect model repo config, chat template, and EOS settings.
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: `harness` | harness:prompt_template, low_hint_overlap | Inspect model repo config, chat template, and EOS settings.
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: `harness` | harness:prompt_template, low_hint_overlap | Inspect model repo config, chat template, and EOS settings.
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: `harness` | harness:prompt_template, low_hint_overlap | Inspect model repo config, chat template, and EOS settings.

### `transformers`

- `microsoft/Florence-2-large-ft`: `harness` | model_error, transformers_model_load_model | Inspect upstream template/tokenizer/config compatibility.
- `prince-canuma/Florence-2-large-ft`: `harness` | model_error, transformers_model_load_model | Inspect upstream template/tokenizer/config compatibility.

## User Buckets

### `recommended`

- `mlx-community/gemma-3-27b-it-qat-4bit`: `clean` | improves trusted hints | utility_delta_positive
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: `clean` | improves trusted hints | utility_delta_positive
- `mlx-community/gemma-3-27b-it-qat-8bit`: `clean` | improves trusted hints | utility_delta_positive

### `caveat`

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: `context_budget` | improves trusted hints | nontext_prompt_burden, utility_delta_positive, metadata_terms
- `mlx-community/InternVL3-14B-8bit`: `clean` | ignores trusted hints | low_hint_overlap
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: `context_budget` | improves trusted hints | nontext_prompt_burden, utility_delta_positive, metadata_terms
- `mlx-community/pixtral-12b-8bit`: `context_budget` | preserves trusted hints | nontext_prompt_burden, trusted_overlap, metadata_terms, context_echo

### `avoid`

- `microsoft/Florence-2-large-ft`: `harness` | preserves trusted hints | model_error, transformers_model_load_model
- `mlx-community/MolmoPoint-8B-fp16`: `harness` | preserves trusted hints | processor_error, model_config_processor_load_processor
- `mlx-community/deepseek-vl2-8bit`: `model_shortcoming` | ignores trusted hints | contract, low_hint_overlap
- `prince-canuma/Florence-2-large-ft`: `harness` | preserves trusted hints | model_error, transformers_model_load_model
- `mlx-community/LFM2-VL-1.6B-8bit`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo
- `mlx-community/nanoLLaVA-1.5-4bit`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, contract, utility:D, trusted_overlap, metadata_terms, fabrication
- `mlx-community/FastVLM-0.5B-bf16`: `model_shortcoming` | ignores trusted hints | contract, low_hint_overlap
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: `harness` | ignores trusted hints | harness:prompt_template, low_hint_overlap
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: `model_shortcoming` | ignores trusted hints | contract, utility:F, low_hint_overlap
- `mlx-community/LFM2.5-VL-1.6B-bf16`: `cutoff` | preserves trusted hints | token_cap, repetitive_tail, abrupt_tail, trusted_overlap, metadata_terms, context_echo
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: `harness` | ignores trusted hints | harness:prompt_template, low_hint_overlap
- `mlx-community/Phi-3.5-vision-instruct-bf16`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, trusted_overlap, metadata_terms, fabrication
- `mlx-community/llava-v1.6-mistral-7b-8bit`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo
- `mlx-community/gemma-3n-E2B-4bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap, metadata_terms
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap
- `HuggingFaceTB/SmolVLM-Instruct`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/SmolVLM-Instruct-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `qnguyen3/nanoLLaVA`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: `harness` | preserves trusted hints | harness:encoding, trusted_overlap, metadata_terms
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: `model_shortcoming` | ignores trusted hints | contract, utility:F, low_hint_overlap
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/gemma-3n-E4B-it-bf16`: `model_shortcoming` | improves trusted hints | metadata_borrowing, contract, utility_delta_positive, metadata_terms
- `microsoft/Phi-3.5-vision-instruct`: `harness` | improves trusted hints | harness:stop_token, abrupt_tail, utility_delta_positive, metadata_terms, degeneration, fabrication
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, contract, trusted_overlap, metadata_terms
- `mlx-community/paligemma2-3b-pt-896-4bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/GLM-4.6V-Flash-mxfp4`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap, reasoning_leak, degeneration
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: `cutoff` | improves trusted hints | token_cap, missing_sections, abrupt_tail, utility_delta_positive, metadata_terms, reasoning_leak, fabrication
- `mlx-community/InternVL3-8B-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, degeneration
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: `cutoff` | improves trusted hints | token_cap, missing_sections, utility_delta_positive, metadata_terms
- `mlx-community/GLM-4.6V-Flash-6bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap, reasoning_leak, degeneration
- `mlx-community/Idefics3-8B-Llama3-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap, instruction_markers, metadata_terms, reasoning_leak
- `Qwen/Qwen3-VL-2B-Instruct`: `cutoff` | preserves trusted hints | token_cap, abrupt_tail, trusted_overlap, metadata_terms, context_echo, fabrication
- `mlx-community/Molmo-7B-D-0924-8bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap
- `mlx-community/X-Reasoner-7B-8bit`: `cutoff` | ignores trusted hints | token_cap, abrupt_tail, low_hint_overlap
- `mlx-community/pixtral-12b-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap
- `mlx-community/GLM-4.6V-nvfp4`: `cutoff` | improves trusted hints | token_cap, missing_sections, abrupt_tail, utility_delta_positive, reasoning_leak
- `mlx-community/Molmo-7B-D-0924-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: `harness` | ignores trusted hints | harness:prompt_template, low_hint_overlap
- `mlx-community/Qwen3.5-35B-A3B-6bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal
- `mlx-community/Qwen3.5-9B-MLX-4bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap
- `mlx-community/Qwen3.5-35B-A3B-bf16`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms, refusal, fabrication
- `mlx-community/Qwen3.5-27B-4bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, trusted_overlap, metadata_terms, refusal, fabrication
- `mlx-community/Qwen3.5-27B-mxfp8`: `cutoff` | preserves trusted hints | token_cap, missing_sections, trusted_overlap, metadata_terms, refusal

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
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model | package=model-config | stage=Processor Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- **Token accounting:** prompt=n/a | text_est=431 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
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
- **Why:** metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=752 | text_est=431 | nontext_est=321 | gen=57 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, contract, utility:D, trusted_overlap, metadata_terms, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords | title words=31 | description sentences=4
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=498 | text_est=431 | nontext_est=67 | gen=191 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** contract, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=502 | text_est=431 | nontext_est=71 | gen=27 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** harness:prompt_template, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1523 | text_est=431 | nontext_est=1092 | gen=8 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** nontext_prompt_burden, utility_delta_positive, metadata_terms
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=caveat | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3089 | text_est=431 | nontext_est=2658 | gen=131 | max=500 | stop=completed
- **Next action:** Reduce prompt/image burden or inspect long-context handling before judging quality.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** contract, utility:F, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1523 | text_est=431 | nontext_est=1092 | gen=23 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, repetitive_tail, abrupt_tail, trusted_overlap, metadata_terms, context_echo
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=204 | keyword duplication=0.93
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=562 | text_est=431 | nontext_est=131 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** harness:prompt_template, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1523 | text_est=431 | nontext_est=1092 | gen=9 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, trusted_overlap, metadata_terms, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=4 | keywords=28
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1315 | text_est=431 | nontext_est=884 | gen=163 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2699 | text_est=431 | nontext_est=2268 | gen=62 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap, metadata_terms
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=759 | text_est=431 | nontext_est=328 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** title words=2
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2281 | text_est=431 | nontext_est=1850 | gen=79 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=607 | text_est=431 | nontext_est=176 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1706 | text_est=431 | nontext_est=1275 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** clean | user=recommended
- **Why:** utility_delta_positive
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=768 | text_est=431 | nontext_est=337 | gen=97 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=recommended
- **Why:** utility_delta_positive
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3090 | text_est=431 | nontext_est=2659 | gen=121 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1706 | text_est=431 | nontext_est=1275 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** context_budget | user=caveat
- **Why:** nontext_prompt_burden, utility_delta_positive, metadata_terms
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=caveat | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3090 | text_est=431 | nontext_est=2659 | gen=134 | max=500 | stop=completed
- **Next action:** Reduce prompt/image burden or inspect long-context handling before judging quality.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=498 | text_est=431 | nontext_est=67 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** context_budget | user=caveat
- **Why:** nontext_prompt_burden, trusted_overlap, metadata_terms, context_echo
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** ok
- **Utility:** user=caveat | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3280 | text_est=431 | nontext_est=2849 | gen=116 | max=500 | stop=completed
- **Next action:** Reduce prompt/image burden or inspect long-context handling before judging quality.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1492 | text_est=431 | nontext_est=1061 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1492 | text_est=431 | nontext_est=1061 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** clean | user=recommended
- **Why:** utility_delta_positive
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=768 | text_est=431 | nontext_est=337 | gen=92 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** harness:encoding, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: description, keywords | title words=55
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2592 | text_est=431 | nontext_est=2161 | gen=89 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** contract, utility:F, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Cambridgeshire, pictured, warm
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=469 | text_est=431 | nontext_est=38 | gen=14 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1492 | text_est=431 | nontext_est=1061 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, contract, utility_delta_positive, metadata_terms
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=767 | text_est=431 | nontext_est=336 | gen=368 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, abrupt_tail, utility_delta_positive, metadata_terms, degeneration, fabrication
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=4 | keywords=54
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=1315 | text_est=431 | nontext_est=884 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, contract, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=468 | text_est=431 | nontext_est=37 | gen=146 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4595 | text_est=431 | nontext_est=4164 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, reasoning_leak, degeneration
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6614 | text_est=431 | nontext_est=6183 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, utility_delta_positive, metadata_terms, reasoning_leak, fabrication
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3371 | text_est=431 | nontext_est=2940 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, degeneration
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2281 | text_est=431 | nontext_est=1850 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, utility_delta_positive, metadata_terms
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title | description sentences=6 | keywords=23 | keyword duplication=0.39
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1812 | text_est=431 | nontext_est=1381 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap, reasoning_leak, degeneration
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6614 | text_est=431 | nontext_est=6183 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2779 | text_est=431 | nontext_est=2348 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap, instruction_markers, metadata_terms, reasoning_leak
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village | nonvisual metadata reused
- **Contract:** missing: title | description sentences=7 | keywords=32 | keyword duplication=0.44
- **Utility:** user=avoid | ignores trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16717 | text_est=431 | nontext_est=16286 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, abrupt_tail, trusted_overlap, metadata_terms, context_echo, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=4 | keywords=63 | keyword duplication=0.59
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16715 | text_est=431 | nontext_est=16284 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap
- **Trusted hints:** preserves trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1677 | text_est=431 | nontext_est=1246 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** keywords=101 | keyword duplication=0.71
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16726 | text_est=431 | nontext_est=16295 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3280 | text_est=431 | nontext_est=2849 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, utility_delta_positive, reasoning_leak
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title, description
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6614 | text_est=431 | nontext_est=6183 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1677 | text_est=431 | nontext_est=1246 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** harness | user=avoid
- **Why:** harness:prompt_template, low_hint_overlap
- **Trusted hints:** ignores trusted hints
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=0 | text_est=431 | nontext_est=0 | gen=0 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16741 | text_est=431 | nontext_est=16310 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: Foxton, Parish, Hall, Annex, village
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16741 | text_est=431 | nontext_est=16310 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms, refusal, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16741 | text_est=431 | nontext_est=16310 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, trusted_overlap, metadata_terms, refusal, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16741 | text_est=431 | nontext_est=16310 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, trusted_overlap, metadata_terms, refusal
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16741 | text_est=431 | nontext_est=16310 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.
