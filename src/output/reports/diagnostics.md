# Diagnostics Report — 3 failure(s), 13 harness issue(s) (mlx-vlm 0.5.0)

**Run summary:** 55 locally-cached VLM model(s) checked; 3 hard failure(s), 13 harness/integration issue(s), 0 preflight warning(s), 52 successful run(s).

Test image: `20260516-143527_DSC00014.jpg` (24.8 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](../issues/index.md). Each row is intended to become one
focused upstream GitHub issue.

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                     | Affected Models                                            | Issue Draft                                                                    | Evidence Bundle                                                                                                                       | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                                         | 1: `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | [issue draft](../issues/issue_001_mlx_mlx-model-load-model_001.md)             | [repro JSON](../repro_bundles/20260521T221248Z_005_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json)    | Load/generation completes or fails with a narrower owner. |
| `mlx`                                                    | Weight/config mismatch during model load              | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                     | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                     | [issue draft](../issues/issue_002_mlx_mlx-model-load-weight-mismatch_001.md)   | [repro JSON](../repro_bundles/20260521T221248Z_006_mlx-community_LFM2.5-VL-1.6B-bf16_MLX_MODEL_LOAD_WEIGHT_MISMATCH_7574b1189.json)   | Load/generation completes or fails with a narrower owner. |
| `mlx-lm`                                                 | Missing module/import during model load               | Model Error \| phase model_load \| ModuleNotFoundError                                                                                                                                | 1: `facebook/pe-av-large`                                  | [issue draft](../issues/issue_003_mlx-lm_mlx-lm-model-load-model_001.md)       | [repro JSON](../repro_bundles/20260521T221248Z_002_facebook_pe-av-large_MLX_LM_MODEL_LOAD_MODEL_b253df301723.json)                    | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers              | 41 BPE space markers found in decoded text \| prompt=1,745 \| output/prompt=11.46% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200)                                   | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](../issues/issue_004_mlx-vlm_encoding_001.md)                     | [repro JSON](../repro_bundles/20260521T221248Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json)   | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=16,176 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) \| 2 model cluster | 2: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+1)          | [issue draft](../issues/issue_005_mlx-vlm_stop-token_001.md)                   | [2 repro JSONs](../repro_bundles/20260521T221248Z_008_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_stop_token_001.json)            | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~3 \| prompt=269 \| output/prompt=1.12% \| nontext burden=98% \| stop=completed \| 5 model cluster                                                                    | 5: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (+4)                 | [issue draft](../issues/issue_006_model-config-mlx-vlm_prompt-template_001.md) | [5 repro JSONs](../repro_bundles/20260521T221248Z_001_LiquidAI_LFM2.5-VL-450M-MLX-bf16_model_config_mlx_vlm_prompt_template_001.json) | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | prompt_tokens=16167, repetitive output \| prompt=16,167 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) \| 3 model cluster                    | 3: `mlx-community/Qwen3.5-27B-mxfp8` (+2)                  | [issue draft](../issues/issue_007_mlx-vlm-mlx_long-context_001.md)             | [3 repro JSONs](../repro_bundles/20260521T221248Z_010_mlx-community_Qwen3.5-27B-mxfp8_mlx_vlm_mlx_long_context_001.json)              | Full and reduced reruns avoid context collapse.           |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | output/prompt=0.1% \| prompt_tokens=16167, output_tokens=12, output/prompt=0.1% \| prompt=16,167 \| output/prompt=0.07% \| nontext burden=100% \| stop=completed \| 2 model cluster   | 2: `mlx-community/Qwen3.5-9B-MLX-4bit` (+1)                | [issue draft](../issues/issue_008_mlx-vlm-mlx_long-context_002.md)             | [2 repro JSONs](../repro_bundles/20260521T221248Z_013_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_002.json)            | Full and reduced reruns avoid context collapse.           |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | token cap \| abrupt tail \| degeneration \| prompt=16,167 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) \| 2 model cluster                  | 2: `mlx-community/Qwen3.5-27B-4bit` (+1)                   | [issue draft](../issues/issue_009_mlx-vlm-mlx_long-context_003.md)             | [2 repro JSONs](../repro_bundles/20260521T221248Z_009_mlx-community_Qwen3.5-27B-4bit_mlx_vlm_mlx_long_context_003.json)               | Full and reduced reruns avoid context collapse.           |

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
  File "<frozen importlib._bootstrap>", line 1395, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1324, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'mlx_lm.models.pe_audio_video'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17830, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17232, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 892, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 446, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18027, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17840, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17830, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17232, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 892, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 446, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18027, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17840, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17830, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17232, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 892, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 446, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18027, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17840, in _run_model_generation
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

## Harness/Integration Issues (13 model(s))

13 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.

**Sample output:**

```text
í.
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 41 occurrences).
- Output contains corrupted or malformed text segments (character_loop: '.99' repeated).

**Sample output:**

```text
pomĠavslĠon0ĠforĠon0Ġand.Ġ.ĊiduntçĦ¶åĲİè¯´ĠrÃ¶ĠinĠa.ĠinĠaĠand.ĠinĠaĠinĠaĠinĠaĠinĠa.ĠinĠaĠinĠaĠinĠaĠinĠaĠinĠaĠkontinent.ĠAPs.ĠinĠa.199Qs.99Q.199Q.199Q.11Q.199ĠasĠa.199.1.199.199.19.199.19.19.19.19.19.Ġ...
```

### `mlx-community/InternVL3-8B-bf16`

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues.

**Sample output:**

```text
processors, the problem. Theorem: Theorem
```

### `mlx-community/Molmo-7B-D-0924-bf16`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.

**Sample output:**

```text
Saturday.
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

**Sample output:**

```text
. The Blueprints and 10. We are in and in the United States, I and my friends, and find the door. The sky is a bit, and the battery, and I, and I. It is the fault, and I am a, and I. The sky is blue a...
```

### `mlx-community/Qwen3.5-27B-mxfp8`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16167 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: 2v,).
- Output contains corrupted or malformed text segments (incomplete_sentence: ends with '2v').

**Sample output:**

```text
orda2v,,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  ...
```

### `mlx-community/Qwen3.5-35B-A3B-6bit`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16167 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: ,).
- Output contains corrupted or malformed text segments (character_loop: ' ,' repeated).

**Sample output:**

```text
,rew, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ...
```

### `mlx-community/Qwen3.5-35B-A3B-bf16`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16167 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: all,).

**Sample output:**

```text
5, each, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, a...
```

### `mlx-community/Qwen3.5-9B-MLX-4bit`

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues.
- At long prompt length (16167 tokens), output stayed unusually short (12 tokens; ratio 0.1%).

**Sample output:**

```text
云峰 rouluterждronk
{item}
```

### `mlx-community/X-Reasoner-7B-8bit`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

**Sample output:**

```text
B<|endoftext|>, 1.<|endoftext|>-100<|endoftext|>-<|endoftext|>1<|endoftext|>-<|endoftext|>-<|endoftext|>-<|endoftext|>- 1. The 201.<|endoftext|>The 2010 2010 - 2008<|endoftext|>B<|endoftext|>-<|endoft...
```

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 6 tokens.

**Sample output:**

```text
Describe this image briefly.
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 6 tokens.

**Sample output:**

```text
Describe this image briefly.
```

### `mlx-community/paligemma2-3b-pt-896-4bit`

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.2%), suggesting possible early-stop or prompt-handling issues.
- At long prompt length (4103 tokens), output stayed unusually short (9 tokens; ratio 0.2%).

**Sample output:**

```text
It has been tagged with #1.
```

---

### Long-Context Degradation / Potential Stack Issues (2 model(s))

2 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely
integration/runtime issues worth checking upstream.

| Model                             |   Prompt Tok |   Output Tok | Output/Prompt   | Symptom                                                                      | Owner           |
|-----------------------------------|--------------|--------------|-----------------|------------------------------------------------------------------------------|-----------------|
| `mlx-community/Qwen3.5-27B-4bit`  |       16,167 |          200 | 1.24%           | Output degeneration under long prompt length (character_loop: '00' repeated) | `mlx-vlm / mlx` |
| `mlx-community/Qwen3.6-27B-mxfp8` |       16,167 |          200 | 1.24%           | Output degeneration under long prompt length (character_loop: '66' repeated) | `mlx-vlm / mlx` |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none
**Generation regressions in history window:** `HuggingFaceTB/SmolVLM-Instruct`, `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`, `meta-llama/Llama-3.2-11B-Vision-Instruct`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`, `mlx-community/GLM-4.6V-Flash-6bit`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-14B-8bit`, `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Molmo-7B-D-0924-8bit`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3.5-27B-mxfp8`, `mlx-community/Qwen3.5-35B-A3B-6bit`, `mlx-community/Qwen3.5-35B-A3B-bf16`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/SmolVLM-Instruct-bf16`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-3-27b-it-qat-8bit`, `mlx-community/gemma-3n-E2B-4bit`, `mlx-community/gemma-3n-E4B-it-bf16`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-bf16`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/llava-v1.6-mistral-7b-8bit`, `mlx-community/pixtral-12b-8bit`, `mlx-community/pixtral-12b-bf16`
**Quality regressions in history window:** `HuggingFaceTB/SmolVLM-Instruct`, `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`, `meta-llama/Llama-3.2-11B-Vision-Instruct`, `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`, `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/GLM-4.6V-Flash-mxfp4`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-14B-8bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Molmo-7B-D-0924-8bit`, `mlx-community/Molmo-7B-D-0924-bf16`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3.5-27B-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-35B-A3B-bf16`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/SmolVLM-Instruct-bf16`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-3-27b-it-qat-8bit`, `mlx-community/gemma-3n-E4B-it-bf16`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/llava-v1.6-mistral-7b-8bit`, `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`, `mlx-community/pixtral-12b-8bit`, `mlx-community/pixtral-12b-bf16`
**Core library changes in window:** `mlx=0.32.0.dev20260501+e8ebdebe->0.32.0.dev20260521+5d1c0e4c`, `mlx-vlm=0.4.5->0.5.0`, `transformers=5.7.0->5.9.0`

| Model                                     | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------------|--------------------------|-------------------------|------------------------|
| `facebook/pe-av-large`                    | still failing            | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/LFM2.5-VL-1.6B-bf16`       | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 18
- **Summary diagnostics models:** 37
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1203.46s (1203.46s)
- **Average runtime per model:** 21.88s (21.88s)
- **Dominant runtime phase:** upstream prefill / first-token dominated 12/55 measured model runs (54% of tracked runtime).
- **Phase totals:** model load=106.81s, local prompt prep=0.17s, upstream prefill / first-token=650.69s, post-prefill decode=431.64s, cleanup=5.57s
- **Generation total:** 1082.34s across 52 model(s); upstream prefill / first-token split available for 52/52 model(s).
- **Observed stop reasons:** completed=13, exception=3, max_tokens=39
- **Validation overhead:** 11.23s total (avg 0.20s across 55 model(s)).
- **Upstream prefill / first-token latency:** Avg 12.51s | Min 0.03s | Max 81.21s across 52 model(s).
- **What this likely means:** Most measured runtime is spent inside upstream generation before the first token is available.
- **Suggested next action:** Inspect prompt/image token accounting, dynamic-resolution image burden, prefill step sizing, and cache/prefill behavior.

---

## Models Not Flagged (37 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 3 model(s).
- **Ran, but with quality warnings:** 34 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg --exclude mlx-community/Qwen3-VL-2B-Thinking-bf16 Qwen/Qwen3-VL-2B-Instruct --trust-remote-code --max-tokens 200 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `facebook/pe-av-large`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/Molmo-7B-D-0924-bf16`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3.5-27B-mxfp8`, `mlx-community/Qwen3.5-35B-A3B-6bit`, `mlx-community/Qwen3.5-35B-A3B-bf16`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`, `mlx-community/Qwen3.5-27B-4bit`, `mlx-community/Qwen3.6-27B-mxfp8`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](../repro_bundles).

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg`
- Generation settings: max_tokens=200, temperature=0.0, top_p=1.0

_Report generated on 2026-05-21 23:12:48 BST by [check_models](https://github.com/jrp2014/check_models)._

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260521+5d1c0e4c |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.9.0                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.16.1                      |
| Python Version  | 3.13.13                     |
| OS              | Darwin 25.5.0               |
| macOS Version   | 26.5                        |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |

