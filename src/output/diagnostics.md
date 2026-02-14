# Diagnostics Report — 3 failure(s), 0 harness issue(s) (mlx-vlm 0.3.12)

## Summary

Automated benchmarking of **43 locally-cached VLM models** found **3 hard failure(s)** and **0 harness/integration issue(s)** in successful models. 40 of 43 models succeeded.

Test image: `20260207-161123_DSC09186.jpg` (27.4 MB).

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.3.12 |
| mlx | 0.30.7.dev20260214+c184262d |
| mlx-lm | 0.30.7 |
| transformers | 5.1.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.4.1 |
| Python Version | 3.13.9 |
| OS | Darwin 25.3.0 |
| macOS Version | 26.3 |
| GPU/Chip | Apple M4 Max |
| GPU Cores | 40 |
| Metal Support | Metal 4 |
| RAM | 128.0 GB |

---

## 1. Weight Mismatch — 1 model(s) [`mlx`] (Priority: Low)

**Error:** `Model loading failed: Missing 1 parameters:`

| Model | Error Stage | Package | Regression vs Prev | First Seen Failing | Recent Repro |
| ----- | ----------- | ------- | ------------------ | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | Weight Mismatch | mlx | no | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

**Traceback (tail):**

```text
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7781, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: Missing 1 parameters: 
language_model.lm_head.weight.
```

<details><summary>Full tracebacks (all models in this cluster)</summary>

### `microsoft/Florence-2-large-ft`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7773, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7739, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 395, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 315, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 1 parameters: 
language_model.lm_head.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7922, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7781, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: Missing 1 parameters: 
language_model.lm_head.weight.
```

</details>

<details><summary>Captured stdout/stderr (all models in this cluster)</summary>

### `microsoft/Florence-2-large-ft`

```text
=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 76398.98it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 2. OOM — 1 model(s) [`mlx`] (Priority: Medium)

**Error:** `Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135699660800 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.`

| Model | Error Stage | Package | Regression vs Prev | First Seen Failing | Recent Repro |
| ----- | ----------- | ------- | ------------------ | ------------------ | ------------ |
| `mlx-community/X-Reasoner-7B-8bit` | OOM | mlx | no | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7841, in _run_model_generation
    raise ValueError(msg) from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135699660800 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

<details><summary>Full tracebacks (all models in this cluster)</summary>

### `mlx-community/X-Reasoner-7B-8bit`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7811, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 599, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 489, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 385, in generate_step
    mx.eval([c.state for c in prompt_cache])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [metal::malloc] Attempting to allocate 135699660800 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7922, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7841, in _run_model_generation
    raise ValueError(msg) from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135699660800 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

</details>

<details><summary>Captured stdout/stderr (all models in this cluster)</summary>

### `mlx-community/X-Reasoner-7B-8bit`

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '0', '7', '-', '1', '6', '1', '1', '2', '3', '_', 'D', 'S', 'C', '0', '9', '1', '8', '6', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Analyze this image and provide structured metadata for image cataloguing.

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

Context: The image is described as ', St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK
St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'.
Existing keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship
Taken on 2026-02-07 16:11:23 GMT (at 16:11:23 local time).

Be factual about visual content.  Include relevant conceptual and emotional keywords where clearly supported by the image.<|im_end|>
<|im_start|>assistant

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 15 files:   0%|          | 0/15 [00:00<?, ?it/s]
Fetching 15 files: 100%|##########| 15/15 [00:00<00:00, 10136.07it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/16869 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/16869 [00:23<?, ?tok/s]
```

</details>

## 3. Model Error — 1 model(s) [`mlx-vlm`] (Priority: Medium)

**Error:** `Model loading failed: RobertaTokenizer has no attribute additional_special_tokens`

| Model | Error Stage | Package | Regression vs Prev | First Seen Failing | Recent Repro |
| ----- | ----------- | ------- | ------------------ | ------------------ | ------------ |
| `prince-canuma/Florence-2-large-ft` | Model Error | mlx-vlm | no | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7781, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

<details><summary>Full tracebacks (all models in this cluster)</summary>

### `prince-canuma/Florence-2-large-ft`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7773, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7739, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 405, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 478, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/molmo/processing_molmo.py", line 758, in _patched_auto_processor_from_pretrained
    return _original_auto_processor_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 355, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/phi3_v/processing_phi3_v.py", line 699, in _patched_auto_processor_from_pretrained
    return _original_auto_processor_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/kimi_vl/processing_kimi_vl.py", line 595, in _patched_auto_processor_from_pretrained
    return _original_auto_processor_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/processing_auto.py", line 394, in from_pretrained
    return processor_class.from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1403, in from_pretrained
    return cls.from_args_and_dict(args, processor_dict, **instantiation_kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1170, in from_args_and_dict
    processor = cls(*args, **valid_kwargs)
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/microsoft/Florence_hyphen_2_hyphen_large_hyphen_ft/4a12a2b54b7016a48a22037fbd62da90cd566f2a/processing_florence2.py", line 87, in __init__
    tokenizer.additional_special_tokens + \
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/tokenization_utils_base.py", line 1291, in __getattr__
    raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")
AttributeError: RobertaTokenizer has no attribute additional_special_tokens. Did you mean: 'add_special_tokens'?

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7922, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7781, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

</details>

<details><summary>Captured stdout/stderr (all models in this cluster)</summary>

### `prince-canuma/Florence-2-large-ft`

```text
=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 82080.31it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
<unknown>:515: SyntaxWarning: invalid escape sequence '\d'
```

</details>

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/gemma-3n-E2B-4bit`, `mlx-community/paligemma2-3b-pt-896-4bit`

| Model | Status vs Previous Run | First Seen Failing | Recent Repro |
| ----- | ---------------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/X-Reasoner-7B-8bit` | still failing | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |
| `prince-canuma/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

---

## Priority Summary

| Priority | Issue | Models Affected | Package |
| -------- | ----- | --------------- | ------- |
| **Low** | Weight Mismatch | 1 (Florence-2-large-ft) | mlx |
| **Medium** | OOM | 1 (X-Reasoner-7B-8bit) | mlx |
| **Medium** | Model Error | 1 (Florence-2-large-ft) | mlx-vlm |

## Reproducibility


```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260207-161123_DSC09186.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose
```


### Target specific failing models

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260207-161123_DSC09186.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
python -m check_models --image /Users/jrp/Pictures/Processed/20260207-161123_DSC09186.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/X-Reasoner-7B-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260207-161123_DSC09186.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models prince-canuma/Florence-2-large-ft
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
Taken on 2026-02-07 16:11:23 GMT (at 16:11:23 local time).

Be factual about visual content.  Include relevant conceptual and emotional keywords where clearly supported by the image.
```

</details>

_Report generated on 2026-02-14 18:19:37 GMT by [check_models](https://github.com/jrp2014/check_models)._
