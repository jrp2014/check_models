<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 3 failure(s), 10 harness issue(s) (mlx-vlm 0.6.1)

**Run summary:** 56 locally-cached VLM model(s) checked; 3 hard failure(s), 10 harness/integration issue(s), 0 preflight warning(s), 53 successful run(s).

Test image: `20260525-164635_DSC00024.jpg` (2.4 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                                                                         | Affected Models                                            | Issue Draft                                                                                                                              | Evidence Bundle                                                                                                                                                                                | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                                                                                             | 1: `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_007_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json)   | Load/generation completes or fails with a narrower owner. |
| `mlx`                                                    | Weight/config mismatch during model load              | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                                                                         | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx_mlx-model-load-weight-mismatch_001.md)   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_008_mlx-community_LFM2.5-VL-1.6B-bf16_MLX_MODEL_LOAD_WEIGHT_MISMATCH_7574b1189.json)  | Load/generation completes or fails with a narrower owner. |
| `mlx-lm`                                                 | Missing module/import during model load               | Model Error \| phase model_load \| ModuleNotFoundError                                                                                                                                                                                    | 1: `facebook/pe-av-large`                                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-lm_mlx-lm-model-load-model_001.md)       | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_001_facebook_pe-av-large_MLX_LM_MODEL_LOAD_MODEL_b253df301723.json)                   | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers              | 96 BPE space markers found in decoded text \| prompt=2,081 \| output/prompt=5.57% \| nontext burden=100% \| stop=completed                                                                                                                | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_encoding_001.md)                     | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json)  | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=770 \| output/prompt=25.97% \| nontext burden=99% \| stop=max_tokens \| hit token cap (200) \| 4 model cluster | 4: `microsoft/Phi-3.5-vision-instruct` (+3)                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm_stop-token_001.md)                   | [4 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_002_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)                 | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~1 \| prompt=266 \| output/prompt=0.38% \| nontext burden=98% \| stop=completed \| 3 model cluster                                                                                                                        | 3: `mlx-community/gemma-3n-E2B-4bit` (+2)                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_model-config-mlx-vlm_prompt-template_001.md) | [3 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_010_mlx-community_gemma-3n-E2B-4bit_model_config_mlx_vlm_prompt_template_001.json) | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | generated_tokens~4 \| prompt_tokens=16250, output_tokens=4, output/prompt=0.0% \| prompt=16,250 \| output/prompt=0.02% \| nontext burden=100% \| stop=completed \| 2 model cluster                                                        | 2: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+1)          | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_007_mlx-vlm-mlx_long-context_001.md)             | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_009_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)     | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18211, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17587, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 894, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 743, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 497, in load_model
    model_config = model_class.ModelConfig.from_dict(config)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/text_only.py", line 39, in from_dict
    model_class, model_args_class = _get_classes(params)
                                    ~~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-lm/mlx_lm/utils.py", line 191, in _get_classes
    raise ValueError(msg)
ValueError: Model type pe_audio_video not supported.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18414, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18221, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type pe_audio_video not supported.

```

<!-- markdownlint-disable MD060 -->

| Model                  | Observed Behavior                                              | First Seen Failing      | Recent Repro           |
|------------------------|----------------------------------------------------------------|-------------------------|------------------------|
| `facebook/pe-av-large` | Model loading failed: Model type pe_audio_video not supported. | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `facebook/pe-av-large`

## 2. Failure affecting 1 model

**Affected models:** `mlx-community/Kimi-VL-A3B-Thinking-8bit`

**Observed error:**

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18211, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17587, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 894, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 743, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 619, in load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18414, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18221, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.

```

<!-- markdownlint-disable MD060 -->

| Model                                     | Observed Behavior                                                                                                                                                                                                     | First Seen Failing      | Recent Repro           |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | Model loading failed: Received 4 parameters not in model: multi_modal_projector.linear_1.biases, multi_modal_projector.linear_1.scales, multi_modal_projector.linear_2.biases, multi_modal_projector.linear_2.scales. | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/Kimi-VL-A3B-Thinking-8bit`

## 3. Failure affecting 1 model

**Affected models:** `mlx-community/LFM2.5-VL-1.6B-bf16`

**Observed error:**

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18211, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17587, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 894, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 743, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 619, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 2 parameters: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18414, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18221, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Missing 2 parameters: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.

```

<!-- markdownlint-disable MD060 -->

| Model                               | Observed Behavior                                                                                                           | First Seen Failing      | Recent Repro           |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16` | Model loading failed: Missing 2 parameters: multi_modal_projector.layer_norm.bias, multi_modal_projector.layer_norm.weight. | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/LFM2.5-VL-1.6B-bf16`

---

## Harness/Integration Issues (10 model(s))

10 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `microsoft/Phi-3.5-vision-instruct`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact, code_snippet).

**Sample output:**

```text
The image captures a dynamic scene of a person riding a green and black jet ski on a body of water. The rider, wearing a black life jacket and gloves, is in motion, with the jet ski creating a spray o...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 96 occurrences).

**Sample output:**

```text
TheĠimageĠdepictsĠaĠpersonĠridingĠaĠjetĠskiĠonĠaĠbodyĠofĠwater.ĠTheĠindividualĠisĠwearingĠaĠblackĠlifeĠjacketĠandĠblackĠgloves,ĠandĠappearsĠtoĠbeĠenjoyingĠtheĠactivity,ĠasĠindicatedĠbyĠtheirĠraisedĠfi...
```

### `mlx-community/GLM-4.6V-Flash-6bit`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

**Sample output:**

```text
<think>Got it, let's see. The image shows a person riding a Sea-Doo jet ski in the ocean. The jet ski is green and yellow with the registration AG-8933. The rider is wearing a life vest, black shirt, ...
```

### `mlx-community/GLM-4.6V-Flash-mxfp4`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

**Sample output:**

```text
<think>Got it, let's see. The image shows a person riding a green and yellow Sea-Doo jet ski on the water. The jet ski has the registration "AG-8933". The person is wearing a black life vest, black sh...
```

### `mlx-community/GLM-4.6V-nvfp4`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'as').
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

**Sample output:**

```text
<think>Okay, let's see. The image shows a young man riding a Sea-Doo jet ski on the, probably, a body of water like the ocean or a lake. The jet ski is black and green with some yellow accents, and it...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 4 tokens.
- At long prompt length (16250 tokens), output stayed unusually short (4 tokens; ratio 0.0%).

**Sample output:**

```text
Mortar
```

### `mlx-community/gemma-3n-E2B-4bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 1 tokens.

**Sample output:**

```text
<empty output>
```

### `mlx-community/gemma-4-31b-bf16`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 1 tokens.

**Sample output:**

```text
<empty output>
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues.

**Sample output:**

```text
A person riding a jet ski in the ocean.
```

### `mlx-community/paligemma2-3b-pt-896-4bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.
- At long prompt length (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%).

**Sample output:**

```text
Jet boat
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none
**Generation regressions in history window:** `mlx-community/GLM-4.6V-nvfp4`
**Quality regressions in history window:** `microsoft/Phi-3.5-vision-instruct`, `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`, `mlx-community/GLM-4.6V-Flash-mxfp4`, `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`, `mlx-community/Qwen3.5-27B-4bit`, `mlx-community/Qwen3.5-35B-A3B-6bit`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/llava-v1.6-mistral-7b-8bit`
**Core library changes in window:** `mlx=0.32.0.dev20260516+7b7c1240->0.32.0.dev20260605+6ea7a00d`, `mlx-vlm=0.5.0->0.6.1`, `transformers=5.8.1->5.10.2`

<!-- markdownlint-disable MD060 -->

| Model                                     | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------------|--------------------------|-------------------------|------------------------|
| `facebook/pe-av-large`                    | still failing            | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/LFM2.5-VL-1.6B-bf16`       | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 13
- **Summary diagnostics models:** 43
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1081.39s (1081.39s)
- **Average runtime per model:** 19.31s (19.31s)
- **Dominant runtime phase:** upstream prefill / first-token dominated 20/56 measured model runs (65% of tracked runtime).
- **Phase totals:** model load=109.95s, local prompt prep=0.18s, upstream prefill / first-token=699.06s, post-prefill decode=262.01s, cleanup=6.07s
- **Generation total:** 961.08s across 53 model(s); upstream prefill / first-token split available for 53/53 model(s).
- **Observed stop reasons:** completed=39, exception=3, max_tokens=14
- **Validation overhead:** 7.04s total (avg 0.13s across 56 model(s)).
- **Upstream prefill / first-token latency:** Avg 13.19s | Min 0.12s | Max 85.90s across 53 model(s).
- **What this likely means:** Most measured runtime is spent inside upstream generation before the first token is available.
- **Suggested next action:** Inspect prompt/image token accounting, dynamic-resolution image burden, prefill step sizing, and cache/prefill behavior.

---

## Models Not Flagged (43 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 30 model(s).
- **Ran, but with quality warnings:** 13 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --exclude mlx-community/Qwen3-VL-2B-Thinking-bf16 Qwen/Qwen3-VL-2B-Instruct --trust-remote-code --max-tokens 200 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `facebook/pe-av-large`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `microsoft/Phi-3.5-vision-instruct`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/GLM-4.6V-Flash-6bit`, `mlx-community/GLM-4.6V-Flash-mxfp4`, `mlx-community/GLM-4.6V-nvfp4`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/gemma-3n-E2B-4bit`, `mlx-community/gemma-4-31b-bf16`, `mlx-community/llava-v1.6-mistral-7b-8bit`, `mlx-community/paligemma2-3b-pt-896-4bit`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg`
- Generation settings: max_tokens=200, temperature=0.0, top_p=1.0

_Report generated on 2026-06-05 13:31:25 BST by [check_models](https://github.com/jrp2014/check_models)._

<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.1                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260605+6ea7a00d                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.3                                                                                                                                                    |
| transformers               | 5.10.2                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.18.0                                                                                                                                                   |
| Python Version             | 3.13.13                                                                                                                                                  |
| OS                         | Darwin 25.5.0                                                                                                                                            |
| macOS Version              | 26.5.1                                                                                                                                                   |
| SDK Version                | 26.5                                                                                                                                                     |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                                       |
| Xcode Version              | 26.5                                                                                                                                                     |
| Xcode Build                | 17F42                                                                                                                                                    |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                               |
| Metal SDK                  | MacOSX26.5.sdk                                                                                                                                           |
| Metal Compiler Version     | Apple metal version 32023.883 (metalfe-32023.883)                                                                                                        |
| Metallib Linker Version    | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                           |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                          |
| GPU/Chip                   | Apple M5 Max                                                                                                                                             |
| GPU Cores                  | 40                                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                                  |
| MLX Install Type           | editable local source                                                                                                                                    |
| MLX Distribution Root      | /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                                       |
| MLX Core Extension         | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (157,751,704 bytes, sha256=e50cd0c2d2ae16781f644476459cbc2ca23b0d428a897140740f86678b5e2bf5) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,675,136 bytes, sha256=7c19dc6a8bf6b56db0155bed7a1d2b0a54182b42eba2fcf0f42b97d3d4c57eff)  |
| RAM                        | 128.0 GB                                                                                                                                                 |
<!-- markdownlint-enable MD060 -->
