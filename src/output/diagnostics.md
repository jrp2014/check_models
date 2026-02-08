# Diagnostics Report — 11 failure(s), 0 harness issue(s) (mlx-vlm 0.3.11)

## Summary

Automated benchmarking of **42 locally-cached VLM models** found **11 hard failure(s)** and **0 harness/integration issue(s)** in successful models. 31 of 42 models succeeded.

Test image: `20260207-161123_DSC09186.jpg` (27.4 MB). All failures are deterministic and reproducible.

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.3.11 |
| mlx | 0.30.7.dev20260208+9cd4b9be |
| mlx-lm | 0.30.6 |
| transformers | 5.1.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.4.1 |
| Python Version | 3.13.9 |
| OS | Darwin 25.2.0 |
| macOS Version | 26.2 |
| GPU/Chip | Apple M4 Max |
| GPU Cores | 40 |
| Metal Support | Metal 4 |
| RAM | 128.0 GB |

---

## 1. Model Error — 4 model(s) [`mlx-vlm`] (Priority: High)

**Error:** `Model generation failed for mlx-community/GLM-4.6V-Flash-6bit: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) cannot be broadcast.`

| Model | Error Stage | Package |
| ----- | ----------- | ------- |
| `mlx-community/GLM-4.6V-Flash-6bit` | Model Error | mlx-vlm |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | Model Error | mlx-vlm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | Model Error | mlx-vlm |
| `mlx-community/paligemma2-3b-pt-896-4bit` | Model Error | mlx-vlm |

**Per-model error messages:**

- `mlx-community/GLM-4.6V-Flash-6bit`: `Model generation failed for mlx-community/GLM-4.6V-Flash-6bit: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) cannot be broadcast.`
- `mlx-community/GLM-4.6V-Flash-mxfp4`: `Model generation failed for mlx-community/GLM-4.6V-Flash-mxfp4: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) cannot be broadcast.`
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: `Model generation failed for mlx-community/Kimi-VL-A3B-Thinking-8bit: [broadcast_shapes] Shapes (999,2048) and (1,0,2048) cannot be broadcast.`
- `mlx-community/paligemma2-3b-pt-896-4bit`: `Model generation failed for mlx-community/paligemma2-3b-pt-896-4bit: [broadcast_shapes] Shapes (1,4,2,2048,4096) and (2048,2048) cannot be broadcast.`

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6950, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/GLM-4.6V-Flash-6bit: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) cannot be broadcast.
```

## 2. Model Error — 2 model(s) [`mlx-vlm`] (Priority: High)

**Error:** `Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Image features and image tokens do not match: tokens: 249, features 266`

| Model | Error Stage | Package |
| ----- | ----------- | ------- |
| `mlx-community/LFM2-VL-1.6B-8bit` | Model Error | mlx-vlm |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | Model Error | mlx-vlm |

**Per-model error messages:**

- `mlx-community/LFM2-VL-1.6B-8bit`: `Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Image features and image tokens do not match: tokens: 249, features 266`
- `mlx-community/LFM2.5-VL-1.6B-bf16`: `Model generation failed for mlx-community/LFM2.5-VL-1.6B-bf16: Image features and image tokens do not match: tokens: 249, features 266`

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6950, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Image features and image tokens do not match: tokens: 249, features 266
```

## 3. Weight Mismatch — 1 model(s) [`mlx`] (Priority: Low)

**Error:** `Model loading failed: Missing 1 parameters:`

| Model | Error Stage | Package |
| ----- | ----------- | ------- |
| `microsoft/Florence-2-large-ft` | Weight Mismatch | mlx |

**Traceback (tail):**

```text
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6895, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: Missing 1 parameters: 
language_model.lm_head.weight.
```

## 4. Model Error — 1 model(s) [`mlx`] (Priority: Medium)

**Error:** `Model loading failed: Expected shape (4096, 4096) but received shape (1024, 4096) for parameter language_model.layers.0.self_attn.k_proj.weight`

| Model | Error Stage | Package |
| ----- | ----------- | ------- |
| `mlx-community/Idefics3-8B-Llama3-bf16` | Model Error | mlx |

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6895, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: Expected shape (4096, 4096) but received shape (1024, 4096) for parameter language_model.layers.0.self_attn.k_proj.weight
```

## 5. OOM — 1 model(s) [`mlx`] (Priority: Medium)

**Error:** `Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135699660800 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.`

| Model | Error Stage | Package |
| ----- | ----------- | ------- |
| `mlx-community/X-Reasoner-7B-8bit` | OOM | mlx |

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6955, in _run_model_generation
    raise ValueError(msg) from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135699660800 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

## 6. No Chat Template — 1 model(s) [`mlx-vlm`] (Priority: Medium)

**Error:** `Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating`

| Model | Error Stage | Package |
| ----- | ----------- | ------- |
| `mlx-community/gemma-3n-E2B-4bit` | No Chat Template | mlx-vlm |

**Traceback (tail):**

```text
    chat_template = self.get_chat_template(chat_template, tools)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/tokenization_utils_base.py", line 3191, in get_chat_template
    raise ValueError(
    ...<4 lines>...
    )
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating
```

## 7. Model Error — 1 model(s) [`mlx-vlm`] (Priority: Medium)

**Error:** `Model loading failed: RobertaTokenizer has no attribute additional_special_tokens`

| Model | Error Stage | Package |
| ----- | ----------- | ------- |
| `prince-canuma/Florence-2-large-ft` | Model Error | mlx-vlm |

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6895, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

---

## Priority Summary

| Priority | Issue | Models Affected | Package |
| -------- | ----- | --------------- | ------- |
| **High** | Model Error | 4 (GLM-4.6V-Flash-6bit, GLM-4.6V-Flash-mxfp4, Kimi-VL-A3B-Thinking-8bit, paligemma2-3b-pt-896-4bit) | mlx-vlm |
| **High** | Model Error | 2 (LFM2-VL-1.6B-8bit, LFM2.5-VL-1.6B-bf16) | mlx-vlm |
| **Low** | Weight Mismatch | 1 (Florence-2-large-ft) | mlx |
| **Medium** | Model Error | 1 (Idefics3-8B-Llama3-bf16) | mlx |
| **Medium** | OOM | 1 (X-Reasoner-7B-8bit) | mlx |
| **Medium** | No Chat Template | 1 (gemma-3n-E2B-4bit) | mlx-vlm |
| **Medium** | Model Error | 1 (Florence-2-large-ft) | mlx-vlm |

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e src/.[dev]

# Run against all cached models
python src/check_models.py

### Target specific failing models:
python src/check_models.py --model microsoft/Florence-2-large-ft
python src/check_models.py --model mlx-community/GLM-4.6V-Flash-6bit
python src/check_models.py --model mlx-community/GLM-4.6V-Flash-mxfp4
```

<details><summary>Prompt used (click to expand)</summary>

```text
Analyze this image and provide structured metadata for image cataloguing.

Respond with exactly these three sections:

Title: A concise, descriptive title (5–12 words).

Description: A factual 1–3 sentence description of what the image shows, covering key subjects, setting, and action.

Keywords: 25–50 comma-separated keywords ordered from most specific to most general, covering:
- Specific subjects (people, objects, animals, landmarks)
- Setting and location (urban, rural, indoor, outdoor)
- Actions and activities
- Concepts and themes (e.g. teamwork, solitude, celebration)
- Mood and emotion (e.g. serene, dramatic, joyful)
- Visual style (e.g. close-up, wide-angle, aerial, silhouette, bokeh)
- Colors and lighting (e.g. golden hour, blue tones, high-key)
- Seasonal and temporal context
- Use-case relevance (e.g. business, travel, food, editorial)

Context: The image is described as ', St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UKSt Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'.
Existing keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship
Taken around 2026-02-07 16:11:23 GMT at 16:11:23.

Be factual about visual content.  Include relevant conceptual and emotional keywords where clearly supported by the image.
```

</details>

_Report generated on 2026-02-08 22:32:28 GMT by [check_models](https://github.com/jrp2014/check_models)._
