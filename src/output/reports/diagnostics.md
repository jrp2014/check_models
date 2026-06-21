<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 2 failure(s), 8 harness issue(s) (mlx-vlm 0.6.3)

**Run summary:** 60 locally-cached VLM model(s) checked; 2 hard failure(s), 8 harness/integration issue(s), 0 preflight warning(s), 58 successful run(s).

Test image: `20260620-195128_DSC00512-Edit.jpg` (48.7 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                                                                          | Evidence Snapshot                                                                                                                                                                                                                           | Affected Models                                            | Issue Draft                                                                                                                              | Evidence Bundle                                                                                                                                                                               | Fixed When                                                |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load                                                         | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                                                                           | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-weight-mismatch_001.md)   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260621T001325Z_004_mlx-community_LFM2.5-VL-1.6B-bf16_MLX_MODEL_LOAD_WEIGHT_MISMATCH_7574b1189.json) | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | mlx-vlm: Model load / model error: property 'eos_token_id' of 'ModelConfig' object has no setter | Model Error \| phase model_load \| AttributeError                                                                                                                                                                                           | 1: `mlx-community/MolmoPoint-8B-fp16`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_mlx-vlm-model-load-model_001.md)     | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260621T001325Z_006_mlx-community_MolmoPoint-8B-fp16_MLX_VLM_MODEL_LOAD_MODEL_7cbd53695717.json)     | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers                                                         | 51 BPE space markers found in decoded text \| prompt=2,622 \| output/prompt=3.17% \| nontext burden=84% \| stop=completed                                                                                                                   | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm_encoding_001.md)                     | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260621T001325Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json) | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text                                                   | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=1,328 \| output/prompt=37.65% \| nontext burden=68% \| stop=max_tokens \| hit token cap (500) \| 2 model cluster | 2: `microsoft/Phi-3.5-vision-instruct` (+1)                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_stop-token_001.md)                   | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260621T001325Z_002_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)                | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                                                            | output/prompt=0.7% \| prompt=1,721 \| output/prompt=0.70% \| nontext burden=75% \| stop=completed \| 3 model cluster                                                                                                                        | 3: `HuggingFaceTB/SmolVLM-Instruct` (+2)                   | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_model-config-mlx-vlm_prompt-template_001.md) | [3 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260621T001325Z_001_HuggingFaceTB_SmolVLM-Instruct_model_config_mlx_vlm_prompt_template_001.json) | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short                                            | prompt_tokens=16736, repetitive output \| prompt=16,736 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) \| 2 model cluster                                                                           | 2: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+1)          | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm-mlx_long-context_001.md)             | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260621T001325Z_007_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)    | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

---

## 1. Failure affecting 1 model

**Affected models:** `mlx-community/LFM2.5-VL-1.6B-bf16`

**Observed error:**

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19006, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18395, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 933, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 783, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 659, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 2 parameters: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19247, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19021, in _run_model_generation
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

## 2. Failure affecting 1 model

**Affected models:** `mlx-community/MolmoPoint-8B-fp16`

**Observed error:**

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19006, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18395, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 933, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 783, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 534, in load_model
    model_config = apply_generation_config_defaults(model_config, config)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 69, in apply_generation_config_defaults
    setattr(model_config, key, config[key])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: property 'eos_token_id' of 'ModelConfig' object has no setter

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19247, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 19021, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter

```

<!-- markdownlint-disable MD060 -->

| Model                              | Observed Behavior                                                                   | First Seen Failing      | Recent Repro           |
|------------------------------------|-------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/MolmoPoint-8B-fp16` | Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/MolmoPoint-8B-fp16`

---

## Harness/Integration Issues (8 model(s))

8 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `HuggingFaceTB/SmolVLM-Instruct`

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues.
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
A tall stone church tower with a clock on it.
```

### `microsoft/Phi-3.5-vision-instruct`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'se').
- Output switched language/script unexpectedly (tokenizer_artifact).

**Sample output:**

```text
Title: St. Peter's Church Tower in Colchester, Essex

Description: The image showcases the tower of St. Peter's Church in Colchester, Essex, captured during the late afternoon with clear skies and sha...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 51 occurrences).
- Output omitted required Title/Description/Keywords sections (description, keywords).

**Sample output:**

```text
Title:ĠStoneĠChurchĠTowerĠwithĠClockĊĊDescription:ĠAĠtallĠstoneĠchurchĠtowerĠwithĠaĠclockĠfaceĠandĠGothic-styleĠwindows,ĠsetĠagainstĠaĠclearĠblueĠskyĠwithĠscatteredĠclouds.ĠTheĠtowerĠisĠbathedĠinĠwarm...
```

### `mlx-community/MiniCPM-V-4.6-8bit`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

**Sample output:**

```text
<think>

</think>

Title:
- Stone church tower with clock and window under blue sky

Description:
- The image shows a stone church tower with a clock and arched window, set against a blue sky with clo...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16736 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: phrase: "structure, architectural detai...").

**Sample output:**

```text
Title: St. Peter's Church, Colchester, England
Description: The exterior of the tower of St. Peter's Church in Colchester, Essex, England, with a clock and stone walls, surrounded by a fence and trees...
```

### `mlx-community/SmolVLM-Instruct-bf16`

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues.
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
A tall stone church tower with a clock on it.
```

### `mlx-community/X-Reasoner-7B-8bit`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16736 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: phrase: "stepped stairway, stepped stai...").

**Sample output:**

```text
Title:
- St Peter's Church Tower, Colchester

Description:
- A stone tower of St Peter's Church, Colchester, stands under a clear blue sky. The tower features a clock face, a pointed window, and a wea...
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 2 tokens.

**Sample output:**

```text
<empty output>
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none
**Generation regressions in history window:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`, `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`
**Quality regressions in history window:** `mlx-community/InternVL3-14B-8bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-3-27b-it-qat-8bit`, `mlx-community/gemma-3n-E4B-it-bf16`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`
**Core library changes in window:** `mlx=0.32.0.dev20260606+8f0e8b14->0.32.0.dev20260621+5abdd04b`, `mlx-vlm=0.6.2->0.6.3`, `transformers=5.10.2->5.12.1`

<!-- markdownlint-disable MD060 -->

| Model                               | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------|--------------------------|-------------------------|------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16` | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16`  | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 10
- **Summary diagnostics models:** 50
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1543.87s (1543.87s)
- **Average runtime per model:** 25.73s (25.73s)
- **Dominant runtime phase:** upstream prefill / first-token dominated 21/60 measured model runs (61% of tracked runtime).
- **Phase totals:** model load=139.64s, local prompt prep=0.21s, upstream prefill / first-token=932.12s, post-prefill decode=441.23s, cleanup=9.77s
- **Generation total:** 1373.35s across 58 model(s); upstream prefill / first-token split available for 58/58 model(s).
- **Observed stop reasons:** completed=42, exception=2, max_tokens=16
- **Validation overhead:** 30.10s total (avg 0.50s across 60 model(s)).
- **Upstream prefill / first-token latency:** Avg 16.07s | Min 0.04s | Max 107.06s across 58 model(s).
- **What this likely means:** Most measured runtime is spent inside upstream generation before the first token is available.
- **Suggested next action:** Inspect prompt/image token accounting, dynamic-resolution image burden, prefill step sizing, and cache/prefill behavior.

---

## Models Not Flagged (50 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 1 model(s).
- **Ran, but with quality warnings:** 49 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260620-195128_DSC00512-Edit.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --presence-context-size 20 --frequency-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/MolmoPoint-8B-fp16`, `HuggingFaceTB/SmolVLM-Instruct`, `microsoft/Phi-3.5-vision-instruct`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/MiniCPM-V-4.6-8bit`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/SmolVLM-Instruct-bf16`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/llava-v1.6-mistral-7b-8bit`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260620-195128_DSC00512-Edit.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-06-21 01:13:25 BST by [check_models](https://github.com/jrp2014/check_models)._

<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.3                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260621+5abdd04b                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
| transformers               | 5.12.1                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.20.1                                                                                                                                                   |
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
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (157,739,544 bytes, sha256=d7cb97dee8940843711ef5df4ff68b80a6e02f98bbd9e085c6794183655c94d1) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,692,160 bytes, sha256=82404f0ce166a876abbb18627634a6fd5361cab3d161c305a7bb36ab54ab086b)  |
| RAM                        | 128.0 GB                                                                                                                                                 |
<!-- markdownlint-enable MD060 -->
