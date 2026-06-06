<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 1 failure(s), 5 harness issue(s) (mlx-vlm 0.6.2)

**Run summary:** 55 locally-cached VLM model(s) checked; 1 hard failure(s), 5 harness/integration issue(s), 0 preflight warning(s), 54 successful run(s).

Test image: `20260525-164635_DSC00024.jpg` (2.4 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                                                                           | Affected Models                                            | Issue Draft                                                                                                                              | Evidence Bundle                                                                                                                                                                                      | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                                                                                               | 1: `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260606T195124Z_001_LiquidAI_LFM2.5-VL-450M-MLX-bf16_MLX_MODEL_LOAD_MODEL_853049863f38.json)                | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers              | 46 BPE space markers found in decoded text \| prompt=2,490 \| output/prompt=3.21% \| nontext burden=86% \| stop=completed                                                                                                                   | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_encoding_001.md)                     | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260606T195124Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json)        | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=1,201 \| output/prompt=41.63% \| nontext burden=71% \| stop=max_tokens \| hit token cap (500) \| 2 model cluster | 2: `microsoft/Phi-3.5-vision-instruct` (+1)                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm_stop-token_001.md)                   | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260606T195124Z_002_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)                       | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | output/prompt=0.4% \| prompt=2,588 \| output/prompt=0.39% \| nontext burden=86% \| stop=completed                                                                                                                                           | 1: `mlx-community/llava-v1.6-mistral-7b-8bit`              | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_model-config-mlx-vlm_prompt-template_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260606T195124Z_007_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json) | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | generated_tokens~4 \| prompt_tokens=16626, output_tokens=4, output/prompt=0.0% \| prompt=16,626 \| output/prompt=0.02% \| nontext burden=98% \| stop=completed                                                                              | 1: `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm-mlx_long-context_001.md)             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260606T195124Z_005_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)              | Full and reduced reruns avoid context collapse.           |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | token cap \| missing sections \| abrupt tail \| prompt=16,639 \| output/prompt=3.00% \| nontext burden=98% \| stop=max_tokens \| hit token cap (500)                                                                                        | 1: `mlx-community/Qwen3.5-35B-A3B-bf16`                    | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm-mlx_long-context_002.md)             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260606T195124Z_006_mlx-community_Qwen3.5-35B-A3B-bf16_mlx_vlm_mlx_long_context_002.json)                   | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

---

## 1. Failure affecting 1 model

**Affected models:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

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
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 756, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 632, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 2 parameters not in model: 
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
ValueError: Model loading failed: Received 2 parameters not in model: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.

```

<!-- markdownlint-disable MD060 -->

| Model                              | Observed Behavior                                                                                                                         | First Seen Failing      | Recent Repro           |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` | Model loading failed: Received 2 parameters not in model: multi_modal_projector.layer_norm.bias, multi_modal_projector.layer_norm.weight. | 2026-05-04 20:21:24 BST | 1/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

---

## Harness/Integration Issues (5 model(s))

5 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `microsoft/Phi-3.5-vision-instruct`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output contains corrupted or malformed text segments (excessive_newlines).
- Output switched language/script unexpectedly (tokenizer_artifact).

**Sample output:**

```text
Title: Man on Jet Ski

Description: A man is riding a green and black jet ski on a body of water, creating a wake behind him.

Keywords: jet ski, water, wake, rider, green, black, motorized, speed, ou...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 46 occurrences).
- Output omitted required Title/Description/Keywords sections (description, keywords).

**Sample output:**

```text
Title:ĊPersonĠridingĠaĠjetĠskiĠonĠwaterĊĊDescription:ĊAĠpersonĠwearingĠaĠblackĠwetsuitĠandĠglovesĠisĠridingĠaĠgreenĠandĠblackĠjetĠskiĠonĠchoppyĠwater,ĠwithĠoneĠarmĠraisedĠinĠcelebration.ĊĊKeywords:Ċje...
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
Young person riding a water sports jet ski on water

Description:
The image shows a person on a jet ski moving through water, with visible splashes and waves. The jet ski is ...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 4 tokens.
- At long prompt length (16626 tokens), output stayed unusually short (4 tokens; ratio 0.0%).

**Sample output:**

```text
Grade 1
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues.
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
The boy is on a jet ski.
```

---

### Long-Context Degradation / Potential Stack Issues (1 model(s))

1 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely
integration/runtime issues worth checking upstream.

<!-- markdownlint-disable MD060 -->

| Model                                |   Prompt Tok |   Output Tok | Output/Prompt   | Symptom                                                                            | Owner           |
|--------------------------------------|--------------|--------------|-----------------|------------------------------------------------------------------------------------|-----------------|
| `mlx-community/Qwen3.5-35B-A3B-bf16` |       16,639 |          500 | 3.00%           | Output degeneration under long prompt length (incomplete_sentence: ends with 'of') | `mlx-vlm / mlx` |
<!-- markdownlint-enable MD060 -->

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none
**Core library changes in window:** `mlx=0.32.0.dev20260516+7b7c1240->0.32.0.dev20260606+8f0e8b14`, `mlx-vlm=0.5.0->0.6.2`, `transformers=5.8.1->5.10.2`

<!-- markdownlint-disable MD060 -->

| Model                              | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|------------------------------------|--------------------------|-------------------------|------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` | new model failing        | 2026-05-04 20:21:24 BST | 1/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 7
- **Summary diagnostics models:** 48
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1379.37s (1379.37s)
- **Average runtime per model:** 25.08s (25.08s)
- **Dominant runtime phase:** upstream prefill / first-token dominated 17/55 measured model runs (48% of tracked runtime).
- **Phase totals:** model load=124.61s, local prompt prep=0.18s, upstream prefill / first-token=664.05s, post-prefill decode=582.51s, cleanup=6.19s
- **Generation total:** 1246.56s across 54 model(s); upstream prefill / first-token split available for 54/54 model(s).
- **Observed stop reasons:** completed=33, exception=1, max_tokens=21
- **Validation overhead:** 6.66s total (avg 0.12s across 55 model(s)).
- **Upstream prefill / first-token latency:** Avg 12.30s | Min 0.07s | Max 78.72s across 54 model(s).
- **What this likely means:** Most measured runtime is spent inside upstream generation before the first token is available.
- **Suggested next action:** Inspect prompt/image token accounting, dynamic-resolution image burden, prefill step sizing, and cache/prefill behavior.

---

## Models Not Flagged (48 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 10 model(s).
- **Ran, but with quality warnings:** 38 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --exclude mlx-community/Qwen3-VL-2B-Thinking-bf16 Qwen/Qwen3-VL-2B-Instruct mlx-community/Qwen3-VL-2B-Instruct-bf16 --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `microsoft/Phi-3.5-vision-instruct`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/MiniCPM-V-4.6-8bit`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/llava-v1.6-mistral-7b-8bit`, `mlx-community/Qwen3.5-35B-A3B-bf16`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-06-06 20:51:24 BST by [check_models](https://github.com/jrp2014/check_models)._

<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.2                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260606+8f0e8b14                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
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
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,675,136 bytes, sha256=8084f4df1833f78e6598f28ff40818d3c7a73343ced3f3c416b192f2292ad933)  |
| RAM                        | 128.0 GB                                                                                                                                                 |
<!-- markdownlint-enable MD060 -->
