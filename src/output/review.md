# Automated Review Digest

_Generated on 2026-04-02 00:00:55 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [check_models.log](check_models.log)

## Maintainer Queue

### `mlx`

- `mlx-community/X-Reasoner-7B-8bit`: `context_budget` | long_context, nontext_prompt_burden, low_hint_overlap | Reduce prompt/image burden or inspect long-context handling before judging quality.
- `Qwen/Qwen3-VL-2B-Instruct`: `cutoff` | token_cap, repetitive_tail, abrupt_tail, trusted_overlap, metadata_terms, fabrication | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: `context_budget` | long_context, nontext_prompt_burden, low_hint_overlap | Reduce prompt/image burden or inspect long-context handling before judging quality.

### `mlx-vlm`

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: `harness` | harness:encoding, utility_delta_negative | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: `harness` | harness:stop_token, missing_sections, abrupt_tail, low_hint_overlap, metadata_terms, reasoning_leak | Inspect prompt-template, stop-token, and decode post-processing behavior.
- `mlx-community/GLM-4.6V-nvfp4`: `harness` | harness:stop_token, utility_delta_positive, metadata_terms, reasoning_leak | Inspect prompt-template, stop-token, and decode post-processing behavior.

### `model`

- `mlx-community/LFM2-VL-1.6B-8bit`: `model_shortcoming` | metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/nanoLLaVA-1.5-4bit`: `cutoff` | token_cap, repetitive_tail, abrupt_tail, trusted_overlap, metadata_terms, context_echo | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: `clean` | low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: `model_shortcoming` | contract, utility:D, low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/FastVLM-0.5B-bf16`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/LFM2.5-VL-1.6B-bf16`: `cutoff` | token_cap, repetitive_tail, trusted_overlap, metadata_terms | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: `clean` | low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: `clean` | utility_delta_positive | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/gemma-3-27b-it-qat-4bit`: `clean` | utility_delta_negative | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/InternVL3-14B-8bit`: `clean` | low_hint_overlap | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: `cutoff` | token_cap, missing_sections, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/pixtral-12b-8bit`: `clean` | utility_delta_negative | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/SmolVLM-Instruct-bf16`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/gemma-3n-E2B-4bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, metadata_terms | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `HuggingFaceTB/SmolVLM-Instruct`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `qnguyen3/nanoLLaVA`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/gemma-3-27b-it-qat-8bit`: `clean` | utility_delta_negative | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/gemma-3n-E4B-it-bf16`: `model_shortcoming` | metadata_borrowing, utility_delta_positive, metadata_terms, fabrication | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: `cutoff` | token_cap, repetitive_tail, trusted_overlap, metadata_terms | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `microsoft/Phi-3.5-vision-instruct`: `cutoff` | token_cap, repetitive_tail, trusted_overlap, metadata_terms | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/paligemma2-3b-pt-896-4bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, unfinished_section, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: `cutoff` | token_cap, missing_sections, utility_delta_positive, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/GLM-4.6V-Flash-6bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, low_hint_overlap, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Idefics3-8B-Llama3-bf16`: `cutoff` | token_cap, missing_sections, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: `cutoff` | token_cap, missing_sections, repetitive_tail, unfinished_section, abrupt_tail, utility_delta_negative | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: `cutoff` | token_cap, unfinished_section, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Molmo-7B-D-0924-8bit`: `model_shortcoming` | metadata_borrowing, utility_delta_positive, metadata_terms | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/InternVL3-8B-bf16`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, degeneration | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: `model_shortcoming` | metadata_borrowing, contract, trusted_overlap, metadata_terms | Treat as a model-quality limitation for this prompt and image.
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: `cutoff` | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/pixtral-12b-bf16`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-35B-A3B-4bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Molmo-7B-D-0924-bf16`: `cutoff` | token_cap, missing_sections, abrupt_tail, low_hint_overlap | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-9B-MLX-4bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-35B-A3B-6bit`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-35B-A3B-bf16`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-27B-4bit`: `cutoff` | token_cap, missing_sections, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal | Inspect token cap and stop behavior before treating this as a model-quality failure.
- `mlx-community/Qwen3.5-27B-mxfp8`: `cutoff` | token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal | Inspect token cap and stop behavior before treating this as a model-quality failure.

### `model-config`

- `mlx-community/MolmoPoint-8B-fp16`: `harness` | processor_error, model_config_processor_load_processor | Inspect model repo config, chat template, and EOS settings.
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: `harness` | harness:prompt_template, low_hint_overlap | Inspect model repo config, chat template, and EOS settings.
- `mlx-community/llava-v1.6-mistral-7b-8bit`: `harness` | harness:prompt_template, low_hint_overlap | Inspect model repo config, chat template, and EOS settings.
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: `harness` | harness:prompt_template, low_hint_overlap | Inspect model repo config, chat template, and EOS settings.

## User Buckets

### `recommended`

- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: `clean` | improves trusted hints | utility_delta_positive

### `caveat`

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: `clean` | ignores trusted hints | low_hint_overlap
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: `clean` | ignores trusted hints | low_hint_overlap
- `mlx-community/InternVL3-14B-8bit`: `clean` | ignores trusted hints | low_hint_overlap
- `mlx-community/X-Reasoner-7B-8bit`: `context_budget` | ignores trusted hints | long_context, nontext_prompt_burden, low_hint_overlap
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: `context_budget` | ignores trusted hints | long_context, nontext_prompt_burden, low_hint_overlap

### `avoid`

- `mlx-community/MolmoPoint-8B-fp16`: `harness` | preserves trusted hints | processor_error, model_config_processor_load_processor
- `mlx-community/LFM2-VL-1.6B-8bit`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: `harness` | ignores trusted hints | harness:prompt_template, low_hint_overlap
- `mlx-community/nanoLLaVA-1.5-4bit`: `cutoff` | preserves trusted hints | token_cap, repetitive_tail, abrupt_tail, trusted_overlap, metadata_terms, context_echo
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: `model_shortcoming` | ignores trusted hints | contract, utility:D, low_hint_overlap
- `mlx-community/FastVLM-0.5B-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/LFM2.5-VL-1.6B-bf16`: `cutoff` | preserves trusted hints | token_cap, repetitive_tail, trusted_overlap, metadata_terms
- `mlx-community/llava-v1.6-mistral-7b-8bit`: `harness` | ignores trusted hints | harness:prompt_template, low_hint_overlap
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: `harness` | ignores trusted hints | harness:prompt_template, low_hint_overlap
- `mlx-community/gemma-3-27b-it-qat-4bit`: `clean` | degrades trusted hints | utility_delta_negative
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap
- `mlx-community/pixtral-12b-8bit`: `clean` | degrades trusted hints | utility_delta_negative
- `mlx-community/SmolVLM-Instruct-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/gemma-3n-E2B-4bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, metadata_terms
- `HuggingFaceTB/SmolVLM-Instruct`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap, degeneration
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `qnguyen3/nanoLLaVA`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: `harness` | degrades trusted hints | harness:encoding, utility_delta_negative
- `mlx-community/gemma-3-27b-it-qat-8bit`: `clean` | degrades trusted hints | utility_delta_negative
- `mlx-community/gemma-3n-E4B-it-bf16`: `model_shortcoming` | improves trusted hints | metadata_borrowing, utility_delta_positive, metadata_terms, fabrication
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap, degeneration
- `mlx-community/Phi-3.5-vision-instruct-bf16`: `cutoff` | preserves trusted hints | token_cap, repetitive_tail, trusted_overlap, metadata_terms
- `microsoft/Phi-3.5-vision-instruct`: `cutoff` | preserves trusted hints | token_cap, repetitive_tail, trusted_overlap, metadata_terms
- `mlx-community/paligemma2-3b-pt-896-4bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, unfinished_section, abrupt_tail, low_hint_overlap
- `mlx-community/GLM-4.6V-Flash-mxfp4`: `harness` | ignores trusted hints | harness:stop_token, missing_sections, abrupt_tail, low_hint_overlap, metadata_terms, reasoning_leak
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: `cutoff` | improves trusted hints | token_cap, missing_sections, utility_delta_positive, reasoning_leak
- `mlx-community/GLM-4.6V-Flash-6bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, low_hint_overlap, reasoning_leak
- `mlx-community/Idefics3-8B-Llama3-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, low_hint_overlap
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: `cutoff` | degrades trusted hints | token_cap, missing_sections, repetitive_tail, unfinished_section, abrupt_tail, utility_delta_negative
- `Qwen/Qwen3-VL-2B-Instruct`: `cutoff` | preserves trusted hints | token_cap, repetitive_tail, abrupt_tail, trusted_overlap, metadata_terms, fabrication
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: `cutoff` | preserves trusted hints | token_cap, unfinished_section, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak
- `mlx-community/Molmo-7B-D-0924-8bit`: `model_shortcoming` | improves trusted hints | metadata_borrowing, utility_delta_positive, metadata_terms
- `mlx-community/InternVL3-8B-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, degeneration
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: `model_shortcoming` | preserves trusted hints | metadata_borrowing, contract, trusted_overlap, metadata_terms
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: `cutoff` | ignores trusted hints | token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- `mlx-community/GLM-4.6V-nvfp4`: `harness` | improves trusted hints | harness:stop_token, utility_delta_positive, metadata_terms, reasoning_leak
- `mlx-community/pixtral-12b-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap
- `mlx-community/Qwen3.5-35B-A3B-4bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal
- `mlx-community/Molmo-7B-D-0924-bf16`: `cutoff` | ignores trusted hints | token_cap, missing_sections, abrupt_tail, low_hint_overlap
- `mlx-community/Qwen3.5-9B-MLX-4bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms
- `mlx-community/Qwen3.5-35B-A3B-6bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal
- `mlx-community/Qwen3.5-35B-A3B-bf16`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo
- `mlx-community/Qwen3.5-27B-4bit`: `cutoff` | preserves trusted hints | token_cap, missing_sections, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal
- `mlx-community/Qwen3.5-27B-mxfp8`: `cutoff` | preserves trusted hints | token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal

## Model Verdicts

### `mlx-community/MolmoPoint-8B-fp16`

- **Verdict:** harness | user=avoid
- **Why:** processor_error, model_config_processor_load_processor
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=model-config | package=model-config | stage=Processor Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/LFM2-VL-1.6B-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=851 | text_est=501 | nontext_est=350 | gen=89 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** harness:prompt_template, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1618 | text_est=501 | nontext_est=1117 | gen=11 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/nanoLLaVA-1.5-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, repetitive_tail, abrupt_tail, trusted_overlap, metadata_terms, context_echo
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=183 | keyword duplication=0.93
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=596 | text_est=501 | nontext_est=95 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

- **Verdict:** clean | user=caveat
- **Why:** low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Handrail, Objects
- **Contract:** title words=4
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3205 | text_est=501 | nontext_est=2704 | gen=97 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** contract, utility:D, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1618 | text_est=501 | nontext_est=1117 | gen=20 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/FastVLM-0.5B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=600 | text_est=501 | nontext_est=99 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/LFM2.5-VL-1.6B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, repetitive_tail, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=23 | keywords=106 | keyword duplication=0.89
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=661 | text_est=501 | nontext_est=160 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- **Verdict:** harness | user=avoid
- **Why:** harness:prompt_template, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=2830 | text_est=501 | nontext_est=2329 | gen=10 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- **Verdict:** harness | user=avoid
- **Why:** harness:prompt_template, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** ok
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model-config | harness=prompt_template
- **Token accounting:** prompt=1618 | text_est=501 | nontext_est=1117 | gen=9 | max=500 | stop=completed
- **Next action:** Inspect model repo config, chat template, and EOS settings.

### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

- **Verdict:** clean | user=caveat
- **Why:** low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Handrail
- **Contract:** ok
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3206 | text_est=501 | nontext_est=2705 | gen=100 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

- **Verdict:** clean | user=recommended
- **Why:** utility_delta_positive
- **Trusted hints:** improves trusted hints
- **Contract:** ok
- **Utility:** user=recommended | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3206 | text_est=501 | nontext_est=2705 | gen=97 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3-27b-it-qat-4bit`

- **Verdict:** clean | user=avoid
- **Why:** utility_delta_negative
- **Trusted hints:** degrades trusted hints
- **Contract:** ok
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=863 | text_est=501 | nontext_est=362 | gen=87 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/InternVL3-14B-8bit`

- **Verdict:** clean | user=caveat
- **Why:** low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** title words=2
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2379 | text_est=501 | nontext_est=1878 | gen=71 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=708 | text_est=501 | nontext_est=207 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/pixtral-12b-8bit`

- **Verdict:** clean | user=avoid
- **Why:** utility_delta_negative
- **Trusted hints:** degrades trusted hints
- **Contract:** description sentences=3
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3399 | text_est=501 | nontext_est=2898 | gen=95 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/SmolVLM-Instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1808 | text_est=501 | nontext_est=1307 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/gemma-3n-E2B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, metadata_terms
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=854 | text_est=501 | nontext_est=353 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `HuggingFaceTB/SmolVLM-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1808 | text_est=501 | nontext_est=1307 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap, degeneration
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1584 | text_est=501 | nontext_est=1083 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1584 | text_est=501 | nontext_est=1083 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `qnguyen3/nanoLLaVA`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=596 | text_est=501 | nontext_est=95 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- **Verdict:** harness | user=avoid
- **Why:** harness:encoding, utility_delta_negative
- **Trusted hints:** degrades trusted hints
- **Contract:** missing: description, keywords | title words=48
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=mlx-vlm | harness=encoding
- **Token accounting:** prompt=2709 | text_est=501 | nontext_est=2208 | gen=75 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/gemma-3-27b-it-qat-8bit`

- **Verdict:** clean | user=avoid
- **Why:** utility_delta_negative
- **Trusted hints:** degrades trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Modern Art, Objects
- **Contract:** ok
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=863 | text_est=501 | nontext_est=362 | gen=82 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/gemma-3n-E4B-it-bf16`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, utility_delta_positive, metadata_terms, fabrication
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=11 | description sentences=9
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=862 | text_est=501 | nontext_est=361 | gen=294 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap, degeneration
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1584 | text_est=501 | nontext_est=1083 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Phi-3.5-vision-instruct-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, repetitive_tail, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=63 | keyword duplication=0.62
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1427 | text_est=501 | nontext_est=926 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `microsoft/Phi-3.5-vision-instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, repetitive_tail, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=63 | keyword duplication=0.62
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1427 | text_est=501 | nontext_est=926 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/paligemma2-3b-pt-896-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, unfinished_section, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=4690 | text_est=501 | nontext_est=4189 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, missing_sections, abrupt_tail, low_hint_overlap, metadata_terms, reasoning_leak
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Handrail, Modern Art, Objects | nonvisual metadata reused
- **Contract:** missing: title, keywords
- **Utility:** user=avoid | ignores trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6707 | text_est=501 | nontext_est=6206 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, utility_delta_positive, reasoning_leak
- **Trusted hints:** improves trusted hints
- **Contract:** missing: title | description sentences=3 | keywords=6
- **Utility:** user=avoid | improves trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3490 | text_est=501 | nontext_est=2989 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/GLM-4.6V-Flash-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, low_hint_overlap, reasoning_leak
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Handrail, Objects, Royston
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=6707 | text_est=501 | nontext_est=6206 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Idefics3-8B-Llama3-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2876 | text_est=501 | nontext_est=2375 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/X-Reasoner-7B-8bit`

- **Verdict:** context_budget | user=caveat
- **Why:** long_context, nontext_prompt_burden, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** title words=3 | description sentences=3
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16824 | text_est=501 | nontext_est=16323 | gen=157 | max=500 | stop=completed
- **Next action:** Reduce prompt/image burden or inspect long-context handling before judging quality.

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, unfinished_section, abrupt_tail, utility_delta_negative
- **Trusted hints:** degrades trusted hints
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | degrades trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1921 | text_est=501 | nontext_est=1420 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `Qwen/Qwen3-VL-2B-Instruct`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, repetitive_tail, abrupt_tail, trusted_overlap, metadata_terms, fabrication
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** title words=2 | keywords=51 | keyword duplication=0.43
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16813 | text_est=501 | nontext_est=16312 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, unfinished_section, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** keywords=19
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16815 | text_est=501 | nontext_est=16314 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Molmo-7B-D-0924-8bit`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, utility_delta_positive, metadata_terms
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** title words=11 | keywords=9
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1783 | text_est=501 | nontext_est=1282 | gen=294 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/InternVL3-8B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap, degeneration
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=2379 | text_est=501 | nontext_est=1878 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `meta-llama/Llama-3.2-11B-Vision-Instruct`

- **Verdict:** model_shortcoming | user=avoid
- **Why:** metadata_borrowing, contract, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=565 | text_est=501 | nontext_est=64 | gen=112 | max=500 | stop=completed
- **Next action:** Treat as a model-quality limitation for this prompt and image.

### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=564 | text_est=501 | nontext_est=63 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/GLM-4.6V-nvfp4`

- **Verdict:** harness | user=avoid
- **Why:** harness:stop_token, utility_delta_positive, metadata_terms, reasoning_leak
- **Trusted hints:** improves trusted hints | nonvisual metadata reused
- **Contract:** missing: title | keywords=54 | keyword duplication=0.35
- **Utility:** user=avoid | improves trusted hints | metadata borrowing
- **Stack / owner:** owner=mlx-vlm | harness=stop_token
- **Token accounting:** prompt=6707 | text_est=501 | nontext_est=6206 | gen=438 | max=500 | stop=completed
- **Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

### `mlx-community/pixtral-12b-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=3399 | text_est=501 | nontext_est=2898 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Molmo-7B-D-0924-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | ignores trusted hints
- **Stack / owner:** owner=model
- **Token accounting:** prompt=1783 | text_est=501 | nontext_est=1282 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, metadata_terms
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: keywords | title words=35 | description sentences=3
- **Utility:** user=avoid | preserves trusted hints | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- **Verdict:** context_budget | user=caveat
- **Why:** long_context, nontext_prompt_burden, low_hint_overlap
- **Trusted hints:** ignores trusted hints | missing terms: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture
- **Contract:** missing: title, description, keywords
- **Utility:** user=caveat | ignores trusted hints
- **Stack / owner:** owner=mlx | harness=long_context
- **Token accounting:** prompt=16824 | text_est=501 | nontext_est=16323 | gen=18 | max=500 | stop=completed
- **Next action:** Reduce prompt/image burden or inspect long-context handling before judging quality.

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3.5-27B-4bit`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

### `mlx-community/Qwen3.5-27B-mxfp8`

- **Verdict:** cutoff | user=avoid
- **Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, refusal
- **Trusted hints:** preserves trusted hints | nonvisual metadata reused
- **Contract:** missing: title, description, keywords
- **Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing
- **Stack / owner:** owner=model
- **Token accounting:** prompt=16839 | text_est=501 | nontext_est=16338 | gen=500 | max=500 | stop=completed
- **Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.
