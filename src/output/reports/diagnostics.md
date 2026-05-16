# Diagnostics Report — 3 failure(s), 14 harness issue(s) (mlx-vlm 0.5.0)

**Run summary:** 55 locally-cached VLM model(s) checked; 3 hard failure(s), 14 harness/integration issue(s), 0 preflight warning(s), 52 successful run(s).

Test image: `20260516-133759_DSC00002_DxO.jpg` (47.6 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                                                       | Affected Models                                            | Issue Draft                                                                                                                              | Evidence Bundle                                                                                                                                                                                | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                                                                           | 1: `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_005_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json)   | Load/generation completes or fails with a narrower owner. |
| `mlx`                                                    | Weight/config mismatch during model load              | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                                                       | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx_mlx-model-load-weight-mismatch_001.md)   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_007_mlx-community_LFM2.5-VL-1.6B-bf16_MLX_MODEL_LOAD_WEIGHT_MISMATCH_7574b1189.json)  | Load/generation completes or fails with a narrower owner. |
| `mlx-lm`                                                 | Missing module/import during model load               | Model Error \| phase model_load \| ModuleNotFoundError                                                                                                                                                                  | 1: `facebook/pe-av-large`                                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-lm_mlx-lm-model-load-model_001.md)       | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_001_facebook_pe-av-large_MLX_LM_MODEL_LOAD_MODEL_b253df301723.json)                   | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers              | 46 BPE space markers found in decoded text \| prompt=2,586 \| output/prompt=19.33% \| nontext burden=84% \| stop=max_tokens \| hit token cap (500)                                                                      | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_encoding_001.md)                     | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_002_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json)  | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;/think&gt; \| prompt_tokens=16730, repetitive output \| prompt=16,730 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) \| 2 model cluster | 2: `mlx-community/Qwen3.6-27B-mxfp8` (+1)                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm_stop-token_001.md)                   | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_015_mlx-community_Qwen3.6-27B-mxfp8_mlx_vlm_stop_token_001.json)                   | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~3 \| prompt=491 \| output/prompt=0.61% \| nontext burden=15% \| stop=completed \| 4 model cluster                                                                                                      | 4: `mlx-community/FastVLM-0.5B-bf16` (+3)                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_model-config-mlx-vlm_prompt-template_001.md) | [4 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_003_mlx-community_FastVLM-0.5B-bf16_model_config_mlx_vlm_prompt_template_001.json) | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | prompt_tokens=16715, repetitive output \| prompt=16,715 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) \| 7 model cluster                                                       | 7: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+6)          | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_007_mlx-vlm-mlx_long-context_001.md)             | [7 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_008_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)     | Full and reduced reruns avoid context collapse.           |

---

## 1. Failure affecting 1 model

**Affected models:** `facebook/pe-av-large`

**Observed error:**

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-lm/mlx_lm/utils.py", line 188, in _get_classes
    arch = importlib.import_module(f"mlx_lm.models.{model_type}")
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/importlib/__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1324, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'mlx_lm.models.pe_audio_video'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17439, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16841, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 443, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 264, in load_model
    model_config = model_class.ModelConfig.from_dict(config)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/text_only.py", line 39, in from_dict
    model_class, model_args_class = _get_classes(params)
                                    ~~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-lm/mlx_lm/utils.py", line 191, in _get_classes
    raise ValueError(msg)
ValueError: Model type pe_audio_video not supported.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17636, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17449, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type pe_audio_video not supported.

```

| Model                  | Observed Behavior                                              | First Seen Failing      | Recent Repro           |
|------------------------|----------------------------------------------------------------|-------------------------|------------------------|
| `facebook/pe-av-large` | Model loading failed: Model type pe_audio_video not supported. | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `facebook/pe-av-large`

## 2. Failure affecting 1 model

**Affected models:** `mlx-community/Kimi-VL-A3B-Thinking-8bit`

**Observed error:**

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17439, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16841, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 443, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 367, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17636, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17449, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.

```

| Model                                     | Observed Behavior                                                                                                                                                                                                     | First Seen Failing      | Recent Repro           |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | Model loading failed: Received 4 parameters not in model: multi_modal_projector.linear_1.biases, multi_modal_projector.linear_1.scales, multi_modal_projector.linear_2.biases, multi_modal_projector.linear_2.scales. | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/Kimi-VL-A3B-Thinking-8bit`

## 3. Failure affecting 1 model

**Affected models:** `mlx-community/LFM2.5-VL-1.6B-bf16`

**Observed error:**

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17439, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16841, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 443, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 367, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 2 parameters: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17636, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17449, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Missing 2 parameters: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.

```

| Model                               | Observed Behavior                                                                                                           | First Seen Failing      | Recent Repro           |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16` | Model loading failed: Missing 2 parameters: multi_modal_projector.layer_norm.bias, multi_modal_projector.layer_norm.weight. | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/LFM2.5-VL-1.6B-bf16`

---

## Harness/Integration Issues (14 model(s))

14 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 46 occurrences).
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output contains corrupted or malformed text segments (character_loop: '11.' repeated).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
odsĠlingeringĠandadĠandĠwithĠaĠveryĠveryĠveryĠveryĠveryĠhighĠand.ad.ĠThisĠisĠaĠveryĠaĠveryĠaĠveryĠaĠveryĠaĠveryĠaĠveryĠa.adĠaĠveryĠa.Ġa.Ġ.Ġ.1Ġa.adĠa.adĠa.ad.ad.ad.Ġ.WeĠareĠa.ad.Ġ.ad.WeĠare.Ġ.ad.WeĠare...
```

### `mlx-community/FastVLM-0.5B-bf16`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).

**Sample output:**

```text
weights:".
```

### `mlx-community/InternVL3-14B-8bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 6 tokens.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).

**Sample output:**

```text
prefacing the same.
```

### `mlx-community/LFM2-VL-1.6B-8bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 6 tokens.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).

**Sample output:**

```text
arak

It is a
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16715 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output became repetitive, indicating possible generation instability (token: phrase: "it is. it is....").
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
The world of the modern, a photo, and the door, 100, I am. A and B, and I, and 3, and 44, 1,0. It's and  here. It's a new, and 1, and 100, I. It and 100, italy, and Europe, 100, and 100, it is. It. It...
```

### `mlx-community/Qwen3.5-27B-4bit`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16730 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output contains corrupted or malformed text segments (character_loop: '，，，' repeated).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
杏花物联的热情和 -，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，...
```

### `mlx-community/Qwen3.5-27B-mxfp8`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16730 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output contains corrupted or malformed text segments (character_loop: '，' repeated).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
欢迎岩：政宗，后现代政治的，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，...
```

### `mlx-community/Qwen3.5-35B-A3B-4bit`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16730 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output became repetitive, indicating possible generation instability (token: 24).
- Output contains corrupted or malformed text segments (character_loop: ' 24' repeated).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24...
```

### `mlx-community/Qwen3.5-35B-A3B-6bit`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16730 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output contains corrupted or malformed text segments (character_loop: '：：' repeated).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
s圈是：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：...
```

### `mlx-community/Qwen3.5-35B-A3B-bf16`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16730 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output contains corrupted or malformed text segments (character_loop: 'red' repeated).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
leredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredred...
```

### `mlx-community/Qwen3.5-9B-MLX-4bit`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16730 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output became repetitive, indicating possible generation instability (token: phrase: "american, t. 0the american,...").
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
前茅要强թ席因天仙牌,
ombeoalibibooks, 0
12The American, T.
0The American, T.
0The American, T.
0The American, T.
0The American, T.
0The American, T.
0The American, T.
0The American, T.
0The American, T.
0The A...
```

### `mlx-community/Qwen3.6-27B-mxfp8`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- At long prompt length (16730 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output became repetitive, indicating possible generation instability (token: phrase: "``` 2500000000 ``` ```...").
- Output formatting deviated from the requested structure. Details: Unknown tags: <think>.
- Output omitted required Title/Description/Keywords sections (title, description, keywords).
- Output leaked reasoning or prompt-template text (<think>).

**Sample output:**

````text
- Neubпіiharuga (2000000000)

<think>

<think>

</think>

```
2500000000
```

```

```

```
2500000000
```

```

```

```
2500000000
```

```

```

```
2500000000
```

```

```

```
2500000000
```

``...
````

### `mlx-community/X-Reasoner-7B-8bit`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|endoftext|&gt; appeared in generated text.
- At long prompt length (16715 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output became repetitive, indicating possible generation instability (token: phrase: "1.<|endoftext|>3 1.<|endoftext...").
- Output switched language/script unexpectedly (tokenizer_artifact).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
1.<|endoftext|>1<|endoftext|>1<|endoftext|>1<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>3 2.<|endoftext|>3 2.<|endoftext|>The 1. The 1.<|endoftext|>The 2. The 1.<|endoftext|>The 1.<|endoftext|>...
```

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
- Do not copy the instructions in the title.
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none

| Model                                     | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------------|--------------------------|-------------------------|------------------------|
| `facebook/pe-av-large`                    | new model failing        | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | new model failing        | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/LFM2.5-VL-1.6B-bf16`       | new model failing        | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 17
- **Summary diagnostics models:** 38
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1605.17s (1605.17s)
- **Average runtime per model:** 29.18s (29.18s)
- **Dominant runtime phase:** post-prefill decode dominated 37/55 measured model runs (49% of tracked runtime).
- **Phase totals:** model load=121.49s, local prompt prep=0.17s, upstream prefill / first-token=686.51s, post-prefill decode=775.37s, cleanup=6.06s
- **Generation total:** 1461.87s across 52 model(s); upstream prefill / first-token split available for 52/52 model(s).
- **Observed stop reasons:** completed=10, exception=3, max_tokens=42
- **Validation overhead:** 18.76s total (avg 0.34s across 55 model(s)).
- **Upstream prefill / first-token latency:** Avg 13.20s | Min 0.05s | Max 88.87s across 52 model(s).
- **What this likely means:** Most measured runtime is spent generating after the first token is available.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (38 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Ran, but with quality warnings:** 38 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260516-133759_DSC00002_DxO.jpg --exclude mlx-community/Qwen3-VL-2B-Thinking-bf16 Qwen/Qwen3-VL-2B-Instruct --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `facebook/pe-av-large`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/InternVL3-14B-8bit`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3.5-27B-4bit`, `mlx-community/Qwen3.5-27B-mxfp8`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-35B-A3B-6bit`, `mlx-community/Qwen3.5-35B-A3B-bf16`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260516-133759_DSC00002_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-05-16 23:30:18 BST by [check_models](https://github.com/jrp2014/check_models)._

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260516+7b7c1240 |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.8.1                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.15.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.5.0               |
| macOS Version   | 26.5                        |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |

