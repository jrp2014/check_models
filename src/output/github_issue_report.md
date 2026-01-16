# MLX VLM Model Compatibility Report — 2026-01-16

## Executive Summary

This report summarizes the results of testing **36 Vision-Language Models** against the `mlx-vlm` 0.3.10, `mlx-lm` 0.30.2, and `mlx` 0.30.4.dev frameworks on Apple Silicon (M4 Max, 128GB RAM, macOS 26.2).

| Metric | Value |
|--------|-------|
| **Total Models Tested** | 36 |
| **Successful** | 28 (78%) |
| **Failed** | 8 (22%) |
| **Average Generation Speed** | 66.4 tok/s |
| **Fastest Model** | mlx-community/LFM2-VL-1.6B-8bit (317 tok/s) |
| **Most Memory Efficient** | mlx-community/LFM2.5-VL-1.6B-bf16 (4.4 GB peak) |

### Test Environment

- **mlx**: 0.30.4.dev20260116+3fe7794f
- **mlx-lm**: 0.30.2
- **mlx-vlm**: 0.3.10
- **transformers**: 5.0.0rc3
- **Python**: 3.13.9
- **Hardware**: Apple M4 Max, 40 GPU cores, 128GB unified memory

---

## 🚨 Critical Issues for Framework Maintainers

### Issue 1: InternVL3 Processor Type Mismatch

**Package**: `mlx-vlm`  
**Affected Model**: `mlx-community/InternVL3-14B-8bit`  
**Severity**: 🔴 High (model completely unusable)

**Error**:

```text
TypeError: Received a InternVLImageProcessor for argument image_processor, 
but a ImageProcessingMixin was expected.
```

**Root Cause**: The custom `InternVLChatProcessor` in `mlx_vlm/models/internvl_chat/processor.py` passes an `InternVLImageProcessor` to the parent `ProcessorMixin.__init__()`, but `transformers` 5.0.0rc3 now performs stricter type checking via `check_argument_for_proper_class()`.

**Suggested Fix**:

```python
# In mlx_vlm/models/internvl_chat/processor.py
# Ensure InternVLImageProcessor inherits from ImageProcessingMixin
# or update the __init__ signature to bypass the type check
```

**Action Required**: Update `InternVLChatProcessor` to be compatible with transformers 5.x type checking.

---

### Issue 2: Kimi-VL Models Missing Import from transformers

**Package**: `mlx-vlm` (upstream model config issue)  
**Affected Models**:

- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`

**Severity**: 🔴 High (models completely unusable)

**Error**:

```text
ImportError: cannot import name '_validate_images_text_input_order' 
from 'transformers.processing_utils'
```

**Root Cause**: The custom `processing_kimi_vl.py` bundled with these models imports `_validate_images_text_input_order` which is a private/internal function that was either removed or renamed in transformers 5.0.0rc3.

**Suggested Fix**:

1. **Short-term**: Pin transformers version or update model's `processing_kimi_vl.py` to not use internal APIs
2. **Long-term**: Open issue with Kimi-VL model maintainers to update their processor code

**Action Required**: Either provide a compatibility shim in mlx-vlm or coordinate with upstream model maintainers.

---

### Issue 3: Florence-2 Weight Mismatch

**Package**: `mlx`  
**Affected Model**: `microsoft/Florence-2-large-ft`

**Severity**: 🟡 Medium

**Error**:

```text
ValueError: Missing 1 parameters: 
language_model.lm_head.weight.
```

**Root Cause**: The model weights file is missing the `language_model.lm_head.weight` parameter. This could be:

1. Incomplete MLX conversion of the original model
2. Weight tying not being handled correctly during conversion

**Suggested Fix**: Re-convert the model with proper weight handling, or implement weight-tying logic in the model loader.

---

### Issue 4: Florence-2 Tokenizer Attribute Error

**Package**: `mlx-vlm`  
**Affected Model**: `prince-canuma/Florence-2-large-ft`

**Severity**: 🟡 Medium

**Error**:

```text
AttributeError: RobertaTokenizer has no attribute additional_special_tokens. 
Did you mean: 'add_special_tokens'?
```

**Root Cause**: The custom `processing_florence2.py` attempts to access `tokenizer.additional_special_tokens` but `RobertaTokenizer` renamed or removed this property.

**Suggested Fix**: Update Florence-2 processor to use the correct tokenizer API:

```python
# Instead of:
tokenizer.additional_special_tokens
# Use:
tokenizer.additional_special_tokens_ids  # or
list(tokenizer.get_added_vocab().keys())
```

---

### Issue 5: Phi-3.5-vision Missing Image Processor File

**Package**: `mlx-vlm`  
**Affected Model**: `mlx-community/Phi-3.5-vision-instruct-bf16`

**Severity**: 🟡 Medium

**Error**:

```text
OSError: does not appear to have a file named image_processing_phi3_v.py
```

**Root Cause**: The MLX-converted model is missing the required `image_processing_phi3_v.py` file that the original Phi-3.5-vision model includes for custom image processing.

**Suggested Fix**:

1. Copy `image_processing_phi3_v.py` from the original `microsoft/Phi-3.5-vision-instruct` model
2. Update the mlx-community conversion script to include processor files

---

### Issue 6: Qwen2-VL OOM on Large Images

**Package**: `mlx`  
**Affected Model**: `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**Severity**: 🟠 High (on large images)

**Error**:

```text
RuntimeError: [metal::malloc] Attempting to allocate 135433060352 bytes 
which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

**Context**: Test image was 5477×7819 pixels (42.8 MPixels). The model attempted to allocate ~126GB for a single buffer.

**Root Cause**: Qwen2-VL's image encoding may be creating very large intermediate tensors for high-resolution images without chunking or tiling.

**Suggested Fix**:

1. Implement automatic image downscaling in mlx-vlm for models with known memory issues
2. Add chunked/tiled image processing for Qwen2-VL
3. Provide better error messaging suggesting image resize

---

### Issue 7: DeepSeek-VL2 Runtime Type Cast Error

**Package**: `mlx`  
**Affected Model**: `mlx-community/deepseek-vl2-8bit`

**Severity**: 🔴 High (model unusable)

**Error**:

```text
RuntimeError: std::bad_cast
```

**Location**:

```python
# In mlx_vlm/models/deepseek_vl_v2/deepseek_vl_v2.py:340
total_tiles.append(pixel_values[idx, : batch_num_tiles[idx]])
```

**Root Cause**: Type mismatch when indexing `pixel_values` tensor, likely due to integer/array type incompatibility in the MLX C++ backend.

**Suggested Fix**: Add explicit type casting or shape validation before the problematic indexing operation.

---

## 📊 Quality Issues in Successful Models

Some models generated output successfully but exhibited quality issues:

| Model | Issues |
|-------|--------|
| `microsoft/Phi-3.5-vision-instruct` | Language mixing, hallucination (generated unrelated content about television history) |
| `Qwen/Qwen3-VL-2B-Instruct` | Verbose (452 tokens), excessive bullet points (22) |
| `mlx-community/SmolVLM-Instruct-bf16` | Verbose (500 tokens), excessive markdown headers |
| `mlx-community/gemma-3-27b-it-qat-4bit` | Excessive bullet points (25) |
| `mlx-community/pixtral-12b-8bit` | Verbose (302 tokens), excessive bullet points (22) |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | Context ignored (only output: "The image is a photograph.") |
| `mlx-community/paligemma2-*` variants | Context ignored (didn't use provided location/date context) |

---

## 🏆 Performance Highlights

### Top 10 Fastest Models (Generation tok/s)

| Rank | Model | Gen TPS | Peak Memory |
|------|-------|---------|-------------|
| 1 | mlx-community/LFM2-VL-1.6B-8bit | 316.8 | 53 GB* |
| 2 | mlx-community/LFM2.5-VL-1.6B-bf16 | 196.6 | 4.4 GB |
| 3 | mlx-community/Ministral-3-3B-Instruct-2512-4bit | 155.9 | 7.9 GB |
| 4 | mlx-community/paligemma2-3b-pt-896-4bit | 126.9 | 8.7 GB |
| 5 | mlx-community/SmolVLM2-2.2B-Instruct-mlx | 116.1 | 5.5 GB |
| 6 | mlx-community/SmolVLM-Instruct-bf16 | 112.8 | 5.5 GB |
| 7 | HuggingFaceTB/SmolVLM-Instruct | 111.8 | 5.5 GB |
| 8 | qnguyen3/nanoLLaVA | 100.0 | 7.8 GB |
| 9 | Qwen/Qwen3-VL-2B-Instruct | 71.9 | 12 GB |
| 10 | mlx-community/Qwen3-VL-2B-Thinking-bf16 | 71.8 | 12 GB |

*Note: LFM2-VL-1.6B-8bit shows 53GB peak but only 1.9GB active memory—investigate potential memory leak/reporting issue.

### Best Quality/Speed Balance

Based on output quality and performance:

1. **mlx-community/SmolVLM2-2.2B-Instruct-mlx** — Fast (116 tok/s), low memory (5.5GB), good output quality
2. **mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit** — Excellent captions, reasonable speed (26 tok/s)
3. **mlx-community/Molmo-7B-D-0924-8bit** — Detailed descriptions (44 tok/s)
4. **mlx-community/gemma-3n-E4B-it-bf16** — Good descriptions, reasonable speed (41.5 tok/s)

---

## 📋 Actionable Items by Package

### For `mlx` maintainers

1. **[BUG]** `std::bad_cast` in tensor indexing (deepseek-vl2)
2. **[BUG]** Weight mismatch handling for Florence-2 (missing lm_head.weight)
3. **[ENHANCEMENT]** Better OOM error messaging with memory usage hints
4. **[INVESTIGATE]** LFM2-VL-1.6B-8bit 53GB peak memory vs 1.9GB active—potential leak?

### For `mlx-vlm` maintainers

1. **[BUG]** InternVL3 processor incompatibility with transformers 5.x
2. **[BUG]** Florence-2 tokenizer API mismatch
3. **[BUG]** Phi-3.5-vision-bf16 missing image_processing_phi3_v.py
4. **[ENHANCEMENT]** Add automatic image downscaling for models with known memory issues
5. **[DOCS]** Document transformers version compatibility requirements

### For Model Uploaders (mlx-community)

1. **Kimi-VL**: Update `processing_kimi_vl.py` to not use internal transformers APIs
2. **Phi-3.5-vision-bf16**: Include `image_processing_phi3_v.py` in model files
3. **Florence-2**: Verify complete weight conversion

---

## 🔧 Suggested Test Matrix

For CI/CD testing, prioritize these models covering different architectures:

| Architecture | Recommended Model | Reason |
|--------------|-------------------|--------|
| SmolVLM | mlx-community/SmolVLM2-2.2B-Instruct-mlx | Fast, reliable |
| Qwen VL | Qwen/Qwen3-VL-2B-Instruct | Popular, good quality |
| LLaVA | qnguyen3/nanoLLaVA | Small, fast |
| Gemma | mlx-community/gemma-3n-E4B-it-bf16 | Recent, well-maintained |
| Molmo | mlx-community/Molmo-7B-D-0924-8bit | Different arch |
| Pixtral | mlx-community/pixtral-12b-8bit | Mistral-based |

---

## Appendix: Full Error Details

<details>
<summary>Click to expand full error tracebacks</summary>

### InternVL3-14B-8bit

```text
File "mlx_vlm/models/internvl_chat/processor.py", line 288, in __init__
    super().__init__(image_processor, tokenizer, chat_template=chat_template)
File "transformers/processing_utils.py", line 614, in __init__
    self.check_argument_for_proper_class(attribute_name, arg)
File "transformers/processing_utils.py", line 697, in check_argument_for_proper_class
    raise TypeError(
        f"Received a {type(argument).__name__} for argument {argument_name}, but a {class_name} was expected."
    )
TypeError: Received a InternVLImageProcessor for argument image_processor, but a ImageProcessingMixin was expected.
```

### Kimi-VL Models

```text
File "processing_kimi_vl.py", line 25, in <module>
    from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order
ImportError: cannot import name '_validate_images_text_input_order' from 'transformers.processing_utils'
```

### DeepSeek-VL2

```text
File "mlx_vlm/models/deepseek_vl_v2/deepseek_vl_v2.py", line 340, in get_input_embeddings
    total_tiles.append(pixel_values[idx, : batch_num_tiles[idx]])
RuntimeError: std::bad_cast
```

### Qwen2-VL OOM

```text
File "mlx_vlm/generate.py", line 332, in generate_step
    mx.async_eval(y)
RuntimeError: [metal::malloc] Attempting to allocate 135433060352 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

### Florence-2 Weight Mismatch

```text
File "mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 1 parameters: 
language_model.lm_head.weight.
```

### Florence-2 Tokenizer Error

```text
File "processing_florence2.py", line 87, in __init__
    tokenizer.additional_special_tokens + \
AttributeError: RobertaTokenizer has no attribute additional_special_tokens. Did you mean: 'add_special_tokens'?
```

### Phi-3.5-vision Missing File

```text
OSError: /Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/... 
does not appear to have a file named image_processing_phi3_v.py.
```

</details>

---

## Full Model Results Table

| Model | Status | Gen TPS | Peak Memory | Quality Issues |
|-------|--------|---------|-------------|----------------|
| HuggingFaceTB/SmolVLM-Instruct | ✅ OK | 111.8 | 5.5 GB | — |
| Qwen/Qwen3-VL-2B-Instruct | ✅ OK | 71.9 | 12 GB | verbose, bullets(22) |
| meta-llama/Llama-3.2-11B-Vision-Instruct | ✅ OK | 3.8 | 25 GB | — |
| microsoft/Florence-2-large-ft | ❌ FAIL | — | — | Weight Mismatch |
| microsoft/Phi-3.5-vision-instruct | ✅ OK | 10.8 | 12 GB | lang_mixing, hallucination |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | ✅ OK | 33.2 | 15 GB | — |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | ✅ OK | 26.0 | 22 GB | — |
| mlx-community/GLM-4.6V-Flash-6bit | ✅ OK | 46.7 | 13 GB | formatting |
| mlx-community/Idefics3-8B-Llama3-bf16 | ✅ OK | 29.3 | 19 GB | formatting |
| mlx-community/InternVL3-14B-8bit | ❌ FAIL | — | — | Processor Error |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | ❌ FAIL | — | — | Lib Version |
| mlx-community/Kimi-VL-A3B-Thinking-8bit | ❌ FAIL | — | — | Lib Version |
| mlx-community/LFM2-VL-1.6B-8bit | ✅ OK | 316.8 | 53 GB | — |
| mlx-community/LFM2.5-VL-1.6B-bf16 | ✅ OK | 196.6 | 4.4 GB | — |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | ✅ OK | 8.3 | 15 GB | — |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | ✅ OK | 155.9 | 7.9 GB | — |
| mlx-community/Molmo-7B-D-0924-8bit | ✅ OK | 43.7 | 41 GB | — |
| mlx-community/Molmo-7B-D-0924-bf16 | ✅ OK | 28.5 | 48 GB | bullets(18) |
| mlx-community/Phi-3.5-vision-instruct-bf16 | ❌ FAIL | — | — | Config Missing |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | ❌ FAIL | — | — | OOM |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | ✅ OK | 71.8 | 12 GB | — |
| mlx-community/SmolVLM-Instruct-bf16 | ✅ OK | 112.8 | 5.5 GB | verbose, formatting |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | ✅ OK | 116.1 | 5.5 GB | — |
| mlx-community/deepseek-vl2-8bit | ❌ FAIL | — | — | Type Cast Error |
| mlx-community/gemma-3-27b-it-qat-4bit | ✅ OK | 27.8 | 19 GB | bullets(25) |
| mlx-community/gemma-3-27b-it-qat-8bit | ✅ OK | 15.3 | 34 GB | bullets(26) |
| mlx-community/gemma-3n-E4B-it-bf16 | ✅ OK | 41.5 | 17 GB | verbose |
| mlx-community/llava-v1.6-mistral-7b-8bit | ✅ OK | 48.0 | 12 GB | context-ignored |
| mlx-community/paligemma2-10b-ft-docci-448-6bit | ✅ OK | 41.4 | 12 GB | context-ignored |
| mlx-community/paligemma2-10b-ft-docci-448-bf16 | ✅ OK | 4.9 | 27 GB | context-ignored |
| mlx-community/paligemma2-3b-ft-docci-448-bf16 | ✅ OK | 17.7 | 11 GB | context-ignored |
| mlx-community/paligemma2-3b-pt-896-4bit | ✅ OK | 126.9 | 8.7 GB | context-ignored |
| mlx-community/pixtral-12b-8bit | ✅ OK | 34.0 | 16 GB | verbose, bullets(22) |
| mlx-community/pixtral-12b-bf16 | ✅ OK | 19.2 | 27 GB | bullets(20) |
| prince-canuma/Florence-2-large-ft | ❌ FAIL | — | — | Model Error |
| qnguyen3/nanoLLaVA | ✅ OK | 100.0 | 7.8 GB | — |

---

*Report generated by check_models v1.0 on 2026-01-16 22:01:14 GMT*  
*Total test runtime: 682.87 seconds*
