# Issue: Weekly Model Performance Report - 12/14/2025

**Status**: 7 Failures | 3 Repetitive Outputs | 1 Hallucination
**Total Models Run**: 35 (28 Successful, 7 Failed)

## 🚨 Critical Failures

The following models failed to run. Stack traces and error details are provided below.

### 1. `microsoft/Florence-2-large-ft`

**Error**: `ValueError: Missing 1 parameters: language_model.lm_head.weight.`
**Analysis**: The model weights appear to be incomplete or the loading logic expects a different structure.

<details>
<summary>Full Traceback</summary>

```
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 4628, in _run_model_generation
    model, tokenizer = load(...)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 302, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 1 parameters: 
language_model.lm_head.weight.
```

</details>

---

### 2. `mlx-community/GLM-4.6V-Flash-6bit`

**Error**: `TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'images'`
**Analysis**: The `transformers` tokenizer is receiving an `images` argument it doesn't support. This suggests a mismatch between `mlx-vlm`'s processing logic and the `transformers` version or the specific tokenizer implementation for GLM-4V.

<details>
<summary>Full Traceback</summary>

```
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 4664, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(...)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 876, in prepare_inputs
    inputs = process_inputs_with_fallback(...)
  File "/transformers/tokenization_utils_fast.py", line 627, in _encode_plus
    batched_output = self._batch_encode_plus(..., **kwargs)
TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'images'
```

</details>

---

### 3. & 4. `mlx-community/Kimi-VL-A3B-Thinking` (2506-bf16 & 8bit)

**Error**: `ImportError: cannot import name '_validate_images_text_input_order' from 'transformers.processing_utils'`
**Analysis**: Both Kimi-VL models failed with the same import error. The code attempts to import a private function `_validate_images_text_input_order` which may have been removed or renamed in the installed version of `transformers` (`4.57.3`).

<details>
<summary>Full Traceback</summary>

```
Traceback (most recent call last):
  File "/processing_kimi_vl.py", line 25, in <module>
    from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order
ImportError: cannot import name '_validate_images_text_input_order' from 'transformers.processing_utils'
```

</details>

---

### 5. `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

**Error**: `ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.`
**Analysis**: The tokenizer configuration for Ministral seems to reference a class that isn't available.

<details>
<summary>Full Traceback</summary>

```
Traceback (most recent call last):
  File "/transformers/models/auto/tokenization_auto.py", line 1153, in from_pretrained
    raise ValueError(f"Tokenizer class {tokenizer_class_candidate} does not exist...")
ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
```

</details>

---

### 6. `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**Error**: `RuntimeError: [metal::malloc] Attempting to allocate 136434163712 bytes...`
**Analysis**: Massive OOM error. The model attempted to allocate ~136GB of memory, which exceeds the buffer limit (~86GB) and physical RAM (128GB). This is unexpected for a "2B" model and 4-bit quantization, suggesting a potential bug in the quantization or model config causing massive activation expansion.

<details>
<summary>Full Traceback</summary>

```
Traceback (most recent call last):
  File "/mlx-vlm/mlx_vlm/generate.py", line 324, in generate_step
    mx.async_eval(y)
RuntimeError: [metal::malloc] Attempting to allocate 136434163712 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

</details>

---

### 7. `mlx-community/gemma-3-12b-pt-8bit`

**Error**: `ValueError: Cannot use apply_chat_template because this processor does not have a chat template.`
**Analysis**: The processor config lacks a chat template, but the generation script attempts to use `apply_chat_template`.

---

## ⚠️ Quality & Performance Observations

### Repetitive Outputs

The following models entered repetition loops:

- **`mlx-community/deepseek-vl2-8bit`**: Repeated `.` indefinitely.
- **`mlx-community/paligemma2-10b-ft-docci`**: Repeated `phrase: "the windows have a..."`.
- **`mlx-community/paligemma2-3b-ft-docci`**: Repeated `phrase: "the window has a..."`.

### Context Ignorance

Several models failed to incorporate the provided context ("Kings Cross Station", "London", "Dec 2025") into their descriptions, treating the image generically:

- `mlx-community/paligemma2-3b-pt-896-4bit`
- `mlx-community/llava-v1.6-mistral-7b-8bit`
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

### Formatting Issues

- **`mlx-community/Idefics3-8B-Llama3-bf16`**: Output contained `<end_of_utterance>` tags.
- **`mlx-community/pixtral-12b`**: Produced excessive bullet points (20+), potentially degrading readability.

### Performance Highlights

- **Fastest Generation**: `prince-canuma/Florence-2-large-ft` (327.6 TPS)
- **Fastest Load**: `HuggingFaceTB/SmolVLM-Instruct` (< 0.00s reported, likely cached/warm)

## Action Plan

1. **Fix Kimi-VL / GLM Compatibility**: Pin `transformers` to a compatible version or patch `mlx-vlm` to handle the missing/changed APIs.
2. **Debug Qwen2-VL OOM**: Investigate why a 2B model is allocating 136GB. Check config values for `vocab_size` or `hidden_size`.
3. **Florence-2 Weights**: Verify the HF repo content for `prince-canuma/Florence-2-large-ft` vs `microsoft/Florence-2-large-ft`.
4. **Gemma-3 Chat Template**: Add a default chat template or fallback logic for Gemma-3 models.
