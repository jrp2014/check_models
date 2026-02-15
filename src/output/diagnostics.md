# Diagnostics Report — 3 failure(s), 6 harness issue(s) (mlx-vlm 0.3.12)

## Summary

Automated benchmarking of **43 locally-cached VLM models** found **3 hard failure(s)** and **6 harness/integration issue(s)** in successful models. 40 of 43 models succeeded.

Test image: `20260214-174320_DSC09249_DxO.jpg` (29.5 MB).

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.3.12 |
| mlx | 0.30.7.dev20260215+c184262d |
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8305, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: Missing 1 parameters: 
language_model.lm_head.weight.
```

<details><summary>Full tracebacks (all models in this cluster)</summary>

### `microsoft/Florence-2-large-ft`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8297, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8263, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8446, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8305, in _run_model_generation
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
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 12104.77it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 2. OOM — 1 model(s) [`mlx`] (Priority: Medium)

**Error:** `Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 136033280000 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.`

| Model | Error Stage | Package | Regression vs Prev | First Seen Failing | Recent Repro |
| ----- | ----------- | ------- | ------------------ | ------------------ | ------------ |
| `mlx-community/X-Reasoner-7B-8bit` | OOM | mlx | no | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8365, in _run_model_generation
    raise ValueError(msg) from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 136033280000 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

<details><summary>Full tracebacks (all models in this cluster)</summary>

### `mlx-community/X-Reasoner-7B-8bit`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8335, in _run_model_generation
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
RuntimeError: [metal::malloc] Attempting to allocate 136033280000 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8446, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8365, in _run_model_generation
    raise ValueError(msg) from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 136033280000 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

</details>

<details><summary>Captured stdout/stderr (all models in this cluster)</summary>

### `mlx-community/X-Reasoner-7B-8bit`

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '1', '4', '-', '1', '7', '4', '3', '2', '0', '_', 'D', 'S', 'C', '0', '9', '2', '4', '9', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata.

Return exactly these three sections:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.

Context: Existing metadata hints (use only if visually consistent):
- Description hint: , Town Centre, Hitchin, England, United Kingdom, UK On a cool February evening in the historic market town of Hitchin, England, the warm glow from a coffee shop illuminates the cobblestone town square against a deep blue twilight sky. Pedestrians and customers are seen enjoying the end of the day, seeking warmth and refreshment. The scene captures the seamless blend of modern life and rich history, with the contem...
- Capture metadata: Taken on 2026-02-14 17:43:20 GMT (at 17:43:20 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.<|im_end|>
<|im_start|>assistant

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 15 files:   0%|          | 0/15 [00:00<?, ?it/s]
Fetching 15 files: 100%|##########| 15/15 [00:00<00:00, 84335.87it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/16560 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/16560 [00:28<?, ?tok/s]
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8305, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

<details><summary>Full tracebacks (all models in this cluster)</summary>

### `prince-canuma/Florence-2-large-ft`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8297, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8263, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8446, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8305, in _run_model_generation
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
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 47127.01it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
<unknown>:515: SyntaxWarning: invalid escape sequence '\d'
```

</details>

---

## Harness/Integration Issues (6 model(s))

These models completed successfully but show integration problems (including empty output, encoding corruption, stop-token leakage, or prompt-template/long-context issues) that indicate stack bugs rather than inherent model quality limits.

### `HuggingFaceTB/SmolVLM-Instruct` — prompt_template

**Tokens:** prompt=1,447, generated=5, ratio=0.35%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(prompt_template), context-ignored

**Details:** output:truncated(5tok)

**Sample output:**

```text
Starbucks.
```

### `Qwen/Qwen3-VL-2B-Instruct` — long_context

**Tokens:** prompt=16,549, generated=500, ratio=3.02%
**Likely package:** `mlx-vlm / mlx`
**Quality flags:** ⚠️harness(long_context), long-context, repetitive(phrase: "town, england, uk, night,...")

**Details:** long_context_repetition(16549tok)

**Sample output:**

```text
Title: Town Centre, Hitchin, England, United Kingdom, UK, evening, cobblestone, coffee shop, people, street, post box, red, Starbucks, nationwide, half-timbered, dark sky

Description: A nighttime vie...
```

### `microsoft/Phi-3.5-vision-instruct` — stop_token

**Tokens:** prompt=1,050, generated=500, ratio=47.62%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(stop_token), lang_mixing

**Details:** token_leak:<|end|>, token_leak:<|endoftext|>, training_leak:code_example

**Sample output:**

```text
Title: Hitchin Town Centre at Dusk

Description: The image captures a serene evening in Hitchin's town centre, with the Starbucks and a traditional half-timbered building illuminated against the eveni...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` — encoding

**Tokens:** prompt=2,166, generated=500, ratio=23.08%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(encoding), generic(0)

**Details:** token_encoding:bpe_space_leak(301)

**Sample output:**

```text
Title:ĠStarbucksĠinĠHistoricĠTownĠSquareĠatĠDuskĊĊDescription:ĠAĠStarbucksĠcoffeeĠshopĠwithĠaĠtimber-framedĠfacadeĠisĠbrightlyĠlitĠinĠaĠcobblestoneĠtownĠsquareĠatĠtwilight.ĠSeveralĠpeopleĠareĠseenĠins...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit` — long_context

**Tokens:** prompt=16,560, generated=500, ratio=3.02%
**Likely package:** `mlx-vlm / mlx`
**Quality flags:** ⚠️harness(long_context), long-context, repetitive(phrase: "market square, coffee shop,...")

**Details:** long_context_repetition(16560tok)

**Sample output:**

```text
Title: "Hitchin's Historic Market Square at Night"
Description: "On a cool February evening, the warm glow from a coffee shop illuminates the cobblestone town square against a deep blue twilight sky. ...
```

### `mlx-community/SmolVLM-Instruct-bf16` — prompt_template

**Tokens:** prompt=1,447, generated=5, ratio=0.35%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(prompt_template), context-ignored

**Details:** output:truncated(5tok)

**Sample output:**

```text
Starbucks.
```

---

## Potential Stack Issues (2 model(s))

These models technically succeeded, but token/output patterns suggest likely integration/runtime issues worth checking upstream.

| Model | Prompt Tok | Output Tok | Output/Prompt | Symptom | Likely Package | Quality Flags |
| ----- | ---------- | ---------- | ------------- | ------- | -------------- | ------------- |
| `Qwen/Qwen3-VL-2B-Instruct` | 16,549 | 500 | 3.02% | Repetition under extreme prompt length | `mlx-vlm / mlx` | ⚠️harness(long_context), long-context, repetitive(phrase: "town, england, uk, night,...") |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | 16,560 | 500 | 3.02% | Repetition under extreme prompt length | `mlx-vlm / mlx` | ⚠️harness(long_context), long-context, repetitive(phrase: "market square, coffee shop,...") |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none

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
| **Medium** | Harness/integration | 6 (SmolVLM-Instruct, Qwen3-VL-2B-Instruct, Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, Qwen2-VL-2B-Instruct-4bit, SmolVLM-Instruct-bf16) | mlx-vlm |

## Reproducibility


```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-174320_DSC09249_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose
```


### Target specific failing models

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-174320_DSC09249_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-174320_DSC09249_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/X-Reasoner-7B-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-174320_DSC09249_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models prince-canuma/Florence-2-large-ft
```

<details><summary>Prompt used (click to expand)</summary>

```text
Analyze this image for cataloguing metadata.

Return exactly these three sections:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.

Context: Existing metadata hints (use only if visually consistent):
- Description hint: , Town Centre, Hitchin, England, United Kingdom, UK On a cool February evening in the historic market town of Hitchin, England, the warm glow from a coffee shop illuminates the cobblestone town square against a deep blue twilight sky. Pedestrians and customers are seen enjoying the end of the day, seeking warmth and refreshment. The scene captures the seamless blend of modern life and rich history, with the contem...
- Capture metadata: Taken on 2026-02-14 17:43:20 GMT (at 17:43:20 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.
```

</details>

_Report generated on 2026-02-15 00:46:38 GMT by [check_models](https://github.com/jrp2014/check_models)._
