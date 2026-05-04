# Diagnostics Report — 3 failure(s), 2 harness issue(s) (mlx-vlm 0.4.5)

**Run summary:** 54 locally-cached VLM model(s) checked; 3 hard failure(s), 2 harness/integration issue(s), 0 preflight warning(s), 51 successful run(s).

Test image: `20260502-173345_DSC09912_DxO.jpg` (27.3 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](../issues/index.md). Each row is intended to become one
focused upstream GitHub issue.

| Target                                         | Problem                                               | Affected Models                                            | Issue Draft                                                                                                    | Evidence Bundle                                                                                                                     | Fixed When                                                |
|------------------------------------------------|-------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                                          | Weight/config mismatch during model load              | 2: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (+1)                 | [issue draft](../issues/issue_001_mlx_mlx-model-load-model_001.md)                                             | [2 repro JSONs](../repro_bundles/20260504T202044Z_001_LiquidAI_LFM2.5-VL-450M-MLX-bf16_MLX_MODEL_LOAD_MODEL_853049863f38.json)      | Load/generation completes or fails with a narrower owner. |
| model configuration / repository               | Processor config is missing image processor           | 1: `mlx-community/MolmoPoint-8B-fp16`                      | [issue draft](../issues/issue_002_model-configuration-repository_model-config-processor-load-processor_001.md) | [repro JSON](../repro_bundles/20260504T202044Z_004_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json)  | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | Tokenizer decode leaked BPE/byte markers              | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](../issues/issue_003_mlx-vlm_encoding_001.md)                                                     | [repro JSON](../repro_bundles/20260504T202044Z_002_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json) | No BPE/byte markers in output.                            |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short | 1: `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [issue draft](../issues/issue_004_mlx-vlm-mlx_long-context_001.md)                                             | [repro JSON](../repro_bundles/20260504T202044Z_005_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)       | Full and reduced reruns avoid context collapse.           |

---

## Upstream Filing Notes

File one upstream issue per row above. The linked issue drafts are the
pasteable bodies; this diagnostics file is the run-level queue and appendix.

- **Issue drafts:** 4 root-cause cluster(s).
- **Suggested targets:** `mlx`=1, `mlx-vlm`=1, `mlx-vlm / mlx`=1, `model configuration/repository`=1.
- **Standalone evidence:** each issue draft includes minimal inline evidence plus an exact cluster rerun command before any appendix detail.
- **Supporting files:** `repro JSON` links point to local repro bundles with prompt, environment, and generated-output context. Attach or publish those JSON files when filing upstream; GitHub will not resolve local artifact paths from a pasted issue body.

## Appendix

The remaining sections keep full run evidence for audit/debugging. They are not intended to be pasted wholesale into an upstream issue.

---

## 1. Failure affecting 2 models

- _Observed:_ Model loading failed: Received 2 parameters not in model:
- _Likely owner:_ `mlx`
- _Why it matters:_ This prevented a complete model response; stage `Model
  Error`; phase `model_load`; code `MLX_MODEL_LOAD_MODEL`; type `ValueError`.
- _Suggested next step:_ check tensor/cache behavior and memory pressure
  handling.
- _Affected models:_ `LiquidAI/LFM2.5-VL-450M-MLX-bf16`,
  `mlx-community/Kimi-VL-A3B-Thinking-8bit`

**Representative maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ runtime_failure \| MLX_MODEL_LOAD_MODEL
- _Summary:_ model error \| mlx model load model
- _Evidence:_ model error \| mlx model load model
- _Token context:_ stop=exception
- _Next action:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.

| Model                                     | Observed Behavior                                         | First Seen Failing      | Recent Repro           |
|-------------------------------------------|-----------------------------------------------------------|-------------------------|------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`        | Model loading failed: Received 2 parameters not in model: | 2026-05-04 20:21:24 BST | 2/2 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | Model loading failed: Received 4 parameters not in model: | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

<details>
<summary>Detailed trace logs (affected models)</summary>

#### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

Traceback tail:

```text
multi_modal_projector.layer_norm.weight.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: Received 2 parameters not in model: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

Captured stdout/stderr:

```text
=== STDERR ===

[21:00:18] ERROR    Failed to load model LiquidAI/LFM2.5-VL-450M-MLX-bf16
                    ╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:17086 in _run_model_generation      │
                    │                                                                                                  │
                    │   17083 │   try:                                                                                 │
                    │   17084 │   │   if phase_timer is not None:                                                      │
                    │   17085 │   │   │   with phase_timer.track("model_load"):                                        │
                    │ ❱ 17086 │   │   │   │   model, processor, config = _load_model(params)                           │
                    │   17087 │   │   else:                                                                            │
                    │   17088 │   │   │   model, processor, config = _load_model(params)                               │
                    │   17089 │   except Exception as load_err:                                                        │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16499 in _load_model                │
                    │                                                                                                  │
                    │   16496 │   │   Tuple of ``(model, processor, config)`` where ``processor`` is an                │
                    │   16497 │   │   ``transformers.ProcessorMixin`` and ``config`` may be ``None``.                  │
                    │   16498 │   """                                                                                  │
                    │ ❱ 16499 │   model, processor = load(                                                             │
                    │   16500 │   │   path_or_hf_repo=params.model_identifier,                                         │
                    │   16501 │   │   adapter_path=params.adapter_path,                                                │
                    │   16502 │   │   lazy=params.lazy,                                                                │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py:412 in load                                 │
                    │                                                                                                  │
                    │    409 │   model_path = get_model_path(                                                          │
                    │    410 │   │   path_or_hf_repo, force_download=force_download, revision=revision                 │
                    │    411 │   )                                                                                     │
                    │ ❱  412 │   model = load_model(model_path, lazy, **kwargs)                                        │
                    │    413 │   if adapter_path is not None:                                                          │
                    │    414 │   │   model = apply_lora_layers(model, adapter_path)                                    │
                    │    415 │   │   model.eval()                                                                      │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py:336 in load_model                           │
                    │                                                                                                  │
                    │    333 │   │   │   )                                                                             │
                    │    334 │   │   model = quantize_activations(model)                                               │
                    │    335 │                                                                                         │
                    │ ❱  336 │   model.load_weights(list(weights.items()))                                             │
                    │    337 │                                                                                         │
                    │    338 │   if not lazy:                                                                          │
                    │    339 │   │   mx.eval(model.parameters())                                                       │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py:185 in load_weights                 │
                    │                                                                                                  │
                    │   182 │   │   │   if extras := (new_weights.keys() - curr_weights.keys()):                       │
                    │   183 │   │   │   │   num_extra = len(extras)                                                    │
                    │   184 │   │   │   │   extras = ",\n".join(sorted(extras))                                        │
                    │ ❱ 185 │   │   │   │   raise ValueError(                                                          │
                    │   186 │   │   │   │   │   f"Received {num_extra} parameters not in model: \n{extras}."           │
                    │   187 │   │   │   │   )                                                                          │
                    │   188 │   │   │   if missing := (curr_weights.keys() - new_weights.keys()):                      │
                    ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
                    ValueError: Received 2 parameters not in model:
                    multi_modal_projector.layer_norm.bias,
                    multi_modal_projector.layer_norm.weight.
[21:00:19] DEBUG    HF Cache Info for LiquidAI/LFM2.5-VL-450M-MLX-bf16: size=860.6 MB, files=13
```

#### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

Traceback tail:

```text
Traceback (most recent call last):
ValueError: Model loading failed: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

Captured stdout/stderr:

```text
=== STDERR ===

[21:04:28] ERROR    Failed to load model mlx-community/Kimi-VL-A3B-Thinking-8bit
                    ╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:17086 in _run_model_generation      │
                    │                                                                                                  │
                    │   17083 │   try:                                                                                 │
                    │   17084 │   │   if phase_timer is not None:                                                      │
                    │   17085 │   │   │   with phase_timer.track("model_load"):                                        │
                    │ ❱ 17086 │   │   │   │   model, processor, config = _load_model(params)                           │
                    │   17087 │   │   else:                                                                            │
                    │   17088 │   │   │   model, processor, config = _load_model(params)                               │
                    │   17089 │   except Exception as load_err:                                                        │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16499 in _load_model                │
                    │                                                                                                  │
                    │   16496 │   │   Tuple of ``(model, processor, config)`` where ``processor`` is an                │
                    │   16497 │   │   ``transformers.ProcessorMixin`` and ``config`` may be ``None``.                  │
                    │   16498 │   """                                                                                  │
                    │ ❱ 16499 │   model, processor = load(                                                             │
                    │   16500 │   │   path_or_hf_repo=params.model_identifier,                                         │
                    │   16501 │   │   adapter_path=params.adapter_path,                                                │
                    │   16502 │   │   lazy=params.lazy,                                                                │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py:412 in load                                 │
                    │                                                                                                  │
                    │    409 │   model_path = get_model_path(                                                          │
                    │    410 │   │   path_or_hf_repo, force_download=force_download, revision=revision                 │
                    │    411 │   )                                                                                     │
                    │ ❱  412 │   model = load_model(model_path, lazy, **kwargs)                                        │
                    │    413 │   if adapter_path is not None:                                                          │
                    │    414 │   │   model = apply_lora_layers(model, adapter_path)                                    │
                    │    415 │   │   model.eval()                                                                      │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py:336 in load_model                           │
                    │                                                                                                  │
                    │    333 │   │   │   )                                                                             │
                    │    334 │   │   model = quantize_activations(model)                                               │
                    │    335 │                                                                                         │
                    │ ❱  336 │   model.load_weights(list(weights.items()))                                             │
                    │    337 │                                                                                         │
                    │    338 │   if not lazy:                                                                          │
                    │    339 │   │   mx.eval(model.parameters())                                                       │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py:185 in load_weights                 │
                    │                                                                                                  │
                    │   182 │   │   │   if extras := (new_weights.keys() - curr_weights.keys()):                       │
                    │   183 │   │   │   │   num_extra = len(extras)                                                    │
                    │   184 │   │   │   │   extras = ",\n".join(sorted(extras))                                        │
                    │ ❱ 185 │   │   │   │   raise ValueError(                                                          │
                    │   186 │   │   │   │   │   f"Received {num_extra} parameters not in model: \n{extras}."           │
                    │   187 │   │   │   │   )                                                                          │
                    │   188 │   │   │   if missing := (curr_weights.keys() - new_weights.keys()):                      │
                    ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
                    ValueError: Received 4 parameters not in model:
                    multi_modal_projector.linear_1.biases,
                    multi_modal_projector.linear_1.scales,
                    multi_modal_projector.linear_2.biases,
                    multi_modal_projector.linear_2.scales.
[21:04:29] DEBUG    HF Cache Info for mlx-community/Kimi-VL-A3B-Thinking-8bit: size=17023.6 MB, files=18
```

</details>

## 2. Failure affecting 1 model

- _Observed:_ Loaded processor has no image_processor; expected multimodal
  processor.
- _Likely owner:_ `model configuration/repository`
- _Why it matters:_ This prevented a complete model response; stage `Processor
  Error`; phase `processor_load`; code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`; type `ValueError`.
- _Suggested next step:_ verify model config, tokenizer files, and revision
  alignment.
- _Affected models:_ `mlx-community/MolmoPoint-8B-fp16`

**Representative maintainer triage:**

- _Likely owner:_ model-config \| confidence=high
- _Classification:_ runtime_failure \| MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- _Summary:_ processor error \| model config processor load processor
- _Evidence:_ processor error \| model config processor load processor
- _Token context:_ stop=exception
- _Next action:_ Inspect the model repo processor/preprocessor config and
  AutoProcessor mapping; the multimodal processor is missing or not exposing
  the image processor expected by mlx-vlm.

| Model                              | Observed Behavior                                                       | First Seen Failing      | Recent Repro           |
|------------------------------------|-------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/MolmoPoint-8B-fp16` | Loaded processor has no image_processor; expected multimodal processor. | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `mlx-community/MolmoPoint-8B-fp16`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/MolmoPoint-8B-fp16`

Traceback tail:

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

Captured stdout/stderr:

```text
=== STDERR ===

/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/modeling_rope_utils.py:1034: FutureWarning: `rope_config_validation` is deprecated and has been removed. Its functionality has been moved to RotaryEmbeddingConfigMixin.validate_rope method. PreTrainedConfig inherits this class, so please call self.validate_rope() instead. Also, make sure to use the new rope_parameters syntax. You can call self.standardize_rope_params() in the meantime.
  warnings.warn(
[21:05:34] ERROR    Model preflight validation failed for mlx-community/MolmoPoint-8B-fp16
                    ╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16857 in _prepare_generation_prompt │
                    │                                                                                                  │
                    │   16854 │   │   )                                                                                │
                    │   16855 │   │   if phase_timer is not None:                                                      │
                    │   16856 │   │   │   with phase_timer.track("prompt_prep"):                                       │
                    │ ❱ 16857 │   │   │   │   _run_model_preflight_validators(                                         │
                    │   16858 │   │   │   │   │   model_identifier=params.model_identifier,                            │
                    │   16859 │   │   │   │   │   processor=processor,                                                 │
                    │   16860 │   │   │   │   │   config=config,                                                       │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16649 in                            │
                    │ _run_model_preflight_validators                                                                  │
                    │                                                                                                  │
                    │   16646 │   │   │   phase="processor_load",                                                      │
                    │   16647 │   │   )                                                                                │
                    │   16648 │   if getattr(processor, "image_processor", None) is None:                              │
                    │ ❱ 16649 │   │   _raise_preflight_error(                                                          │
                    │   16650 │   │   │   "Loaded processor has no image_processor; expected multimodal processor.",   │
                    │   16651 │   │   │   phase="processor_load",                                                      │
                    │   16652 │   │   )                                                                                │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16582 in _raise_preflight_error     │
                    │                                                                                                  │
                    │   16579                                                                                          │
                    │   16580 def _raise_preflight_error(message: str, *, phase: str) -> NoReturn:                     │
                    │   16581 │   """Raise a preflight ValueError annotated with the failing phase."""                 │
                    │ ❱ 16582 │   raise _tag_exception_failure_phase(ValueError(message), phase)                       │
                    │   16583                                                                                          │
                    │   16584                                                                                          │
                    │   16585 def _validate_model_artifact_layout(                                                     │
                    ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
                    ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

## Harness/Integration Issues (2 model(s))

2 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Observed:_ Decoded output contains tokenizer artifacts that should not
  appear in user-facing text.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=3,619, output=108, output/prompt=2.98%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| encoding
- _Summary:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 61 occurrences). \| nontext prompt burden=88% \| missing
  sections: description, keywords \| missing terms: vast, expanse, adorned,
  small, floats
- _Evidence:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 61 occurrences).
- _Token context:_ prompt=3,619 \| output/prompt=2.98% \| nontext burden=88%
  \| stop=completed
- _Next action:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 61 occurrences).
- Output omitted required Title/Description/Keywords sections (description, keywords).

**Sample output:**

```text
Title:ĠClassicĠsailboatĠmooredĠinĠestuaryĊĊDescription:ĠAĠclassic-styleĠsailboatĠwithĠaĠdarkĠhullĠandĠwoodenĠmastĠisĠmooredĠinĠaĠcalmĠestuaryĠduringĠlowĠtide.ĠTheĠwaterĠhasĠreceded,ĠexposingĠgreen,Ġal...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=16,901, output=11, output/prompt=0.07%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ context_budget \| long_context
- _Summary:_ Output is very short relative to prompt size (0.1%), suggesting
  possible early-stop or prompt-handling issues. \| At long prompt length
  (16901 tokens), output stayed unusually short (11 tokens; ratio 0.1%). \|
  output/prompt=0.07% \| nontext prompt burden=97%
- _Evidence:_ Output is very short relative to prompt size (0.1%), suggesting
  possible early-stop or prompt-handling issues. \| At long prompt length
  (16901 tokens), output stayed unusually short (11 tokens; ratio 0.1%).
- _Token context:_ prompt=16,901 \| output/prompt=0.07% \| nontext burden=97%
  \| stop=completed
- _Next action:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues.
- At long prompt length (16901 tokens), output stayed unusually short (11 tokens; ratio 0.1%).
- Model output may not follow prompt or image contents (missing: classic, style, sailboat, dark, hull).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
Boat, boat, boat, boat, boat
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none

| Model                                     | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------------|--------------------------|-------------------------|------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`        | still failing            | 2026-05-04 20:21:24 BST | 2/2 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | still failing            | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16`        | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 5
- **Summary diagnostics models:** 49
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1221.63s (1221.63s)
- **Average runtime per model:** 22.62s (22.62s)
- **Dominant runtime phase:** upstream prefill / first-token dominated 16/54 measured model runs (52% of tracked runtime).
- **Phase totals:** model load=106.69s, local prompt prep=0.16s, upstream prefill / first-token=627.80s, post-prefill decode=472.75s, cleanup=5.56s
- **Generation total:** 1100.55s across 51 model(s); upstream prefill / first-token split available for 51/51 model(s).
- **Observed stop reasons:** completed=51, exception=3
- **Validation overhead:** 11.21s total (avg 0.21s across 54 model(s)).
- **Upstream prefill / first-token latency:** Avg 12.31s | Min 0.06s | Max 80.88s across 51 model(s).
- **What this likely means:** Most measured runtime is spent inside upstream generation before the first token is available.
- **Suggested next action:** Inspect prompt/image token accounting, dynamic-resolution image burden, prefill step sizing, and cache/prefill behavior.

---

## Models Not Flagged (49 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Clean output (7 model(s))

- `mlx-community/InternVL3-8B-bf16`
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`
- `mlx-community/gemma-3-27b-it-qat-8bit`
- `mlx-community/gemma-4-26b-a4b-it-4bit`
- `mlx-community/gemma-4-31b-it-4bit`

### Ran, but with quality warnings (42 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Output omitted required Title/Description/Keywords sections (title).
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `microsoft/Phi-3.5-vision-instruct`: Output became repetitive, indicating possible generation instability (token: phrase: "flags, flags, flags, flags,...").
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections (title, description).
- `mlx-community/FastVLM-0.5B-bf16`: Output omitted required Title/Description/Keywords sections (keywords).
- `mlx-community/GLM-4.6V-Flash-6bit`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/GLM-4.6V-nvfp4`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Output formatting deviated from the requested structure. Details: Unknown tags: <end_of_utterance>.
- `mlx-community/InternVL3-14B-8bit`: Output appears to copy prompt context verbatim (46% overlap).
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Output omitted required Title/Description/Keywords sections (title).
- `mlx-community/LFM2-VL-1.6B-8bit`: Output appears to copy prompt context verbatim (64% overlap).
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "mudflats, flags, boat, water,.....
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Molmo-7B-D-0924-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Molmo-7B-D-0924-bf16`: Model refused or deflected the requested task (insufficient_info).
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "flags, flags, flags, flags,...").
- `mlx-community/Qwen3.5-27B-4bit`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-27B-mxfp8`: Keyword count violation (20; expected 10-18)
- `mlx-community/Qwen3.5-35B-A3B-4bit`: Output omitted required Title/Description/Keywords sections (description, keywords).
- `mlx-community/Qwen3.5-35B-A3B-6bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.5-35B-A3B-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.5-9B-MLX-4bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.6-27B-mxfp8`: Title length violation (4 words; expected 5-10)
- `mlx-community/SmolVLM-Instruct-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Output appears to copy prompt context verbatim (66% overlap).
- `mlx-community/X-Reasoner-7B-8bit`: Description sentence violation (3; expected 1-2)
- `mlx-community/gemma-3-27b-it-qat-4bit`: Keyword count violation (19; expected 10-18)
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents (missing: classic, style, sailboat, dark, hull).
- `mlx-community/gemma-3n-E4B-it-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/gemma-4-31b-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/llava-v1.6-mistral-7b-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/nanoLLaVA-1.5-4bit`: Output omitted required Title/Description/Keywords sections (keywords).
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: Model output may not follow prompt or image contents (missing: classic, style, sailboat, dark, hull).
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: Model output may not follow prompt or image contents (missing: classic, style, sailboat, dark, hull).
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: Model output may not follow prompt or image contents (missing: classic, style, sailboat, dark, hull).
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents (missing: classic, style, sailboat, dark, hull).
- `mlx-community/pixtral-12b-8bit`: Output appears to copy prompt context verbatim (58% overlap).
- `mlx-community/pixtral-12b-bf16`: Output appears to copy prompt context verbatim (56% overlap).
- `qnguyen3/nanoLLaVA`: Output omitted required Title/Description/Keywords sections (keywords).

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260504+e8ebdebe |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.8.0.dev0                  |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.13.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.4.0               |
| macOS Version   | 26.4.1                      |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg --exclude mlx-community/Qwen3-VL-2B-Thinking-bf16 Qwen/Qwen3-VL-2B-Instruct --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

### Portable upstream probes (no local image required)

These probes are not a substitute for the original repro bundle. They help
upstream maintainers separate package/import problems, model
repository/config/load problems, and image-dependent generation problems
without needing the original local image.

```bash
# 1) Environment sanity: package versions + import smoke test
python -m pip show mlx mlx-vlm mlx-lm transformers huggingface-hub tokenizers

python - <<'PY'
import importlib
packages = ('mlx', 'mlx_vlm', 'mlx_lm', 'transformers', 'huggingface_hub', 'tokenizers')
for name in packages:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, '__version__', 'unknown')
        print(f'{name} OK {version}')
    except Exception as exc:
        print(f'{name} FAIL {type(exc).__name__}: {exc}')
PY

# 2) Model load/config probe: no original image required
python - <<'PY'
from mlx_vlm.utils import load  # mlx_vlm.utils.load

MODELS = ['LiquidAI/LFM2.5-VL-450M-MLX-bf16', 'mlx-community/Kimi-VL-A3B-Thinking-8bit', 'mlx-community/MolmoPoint-8B-fp16', 'mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit', 'mlx-community/Qwen2-VL-2B-Instruct-4bit']
LOAD_KWARGS = {'adapter_path': None, 'lazy': False, 'revision': None, 'trust_remote_code': True}

for model_id in MODELS:
    print(f"== {model_id} ==")
    try:
        model, processor = load(model_id, **LOAD_KWARGS)
    except Exception as exc:
        print(f"load FAIL: {type(exc).__name__}: {exc}")
        continue

    config = getattr(model, 'config', None)
    tokenizer = getattr(processor, 'tokenizer', None)
    image_processor = getattr(processor, 'image_processor', None)
    print("load OK")
    print(f"model_class={type(model).__name__}")
    config_class = type(config).__name__ if config is not None else 'None'
    print(f"config_class={config_class}")
    model_type = getattr(config, 'model_type', None)
    print(f"model_type={model_type}")
    print(f"processor_class={type(processor).__name__}")
    tokenizer_class = type(tokenizer).__name__ if tokenizer is not None else 'None'
    print(f"tokenizer_class={tokenizer_class}")
    print(f"has_image_processor={image_processor is not None}")
    eos_token = getattr(tokenizer, 'eos_token', None)
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    print(f"eos_token={eos_token}")
    print(f"eos_token_id={eos_token_id}")
PY

# 3) Synthetic-image generation probe: separates image-dependent failures
python - <<'PY'
from PIL import Image

size = 32
image = Image.new('RGB', (size, size))
pixels = [
    ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256)
    for y in range(size)
    for x in range(size)
]
image.putdata(pixels)
image.save('./check_models_portable_probe.png')
print('wrote ./check_models_portable_probe.png')
PY

python -m check_models --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --prompt 'Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: A classic-style sailboat with a dark hull and wooden mast is moored in a calm estuary during low tide. The water has receded, exposing a vast expanse of green, algae-covered mudflats behind the vessel. The boat, adorned with a string of small flags, floats peacefully, waiting for the tide to rise again.
- Capture metadata: Taken on 2026-05-02 18:33:45 BST (at 18:33:45 local time). GPS: 52.089294°N, 1.317741°E.' --image ./check_models_portable_probe.png --models LiquidAI/LFM2.5-VL-450M-MLX-bf16 mlx-community/Kimi-VL-A3B-Thinking-8bit mlx-community/MolmoPoint-8B-fp16 mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit mlx-community/Qwen2-VL-2B-Instruct-4bit
```

### Target specific failing models

**Note:** A comprehensive JSON reproduction bundle including system info and
the exact prompt trace has been exported to
[repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles)
for each failing model.

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models LiquidAI/LFM2.5-VL-450M-MLX-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Kimi-VL-A3B-Thinking-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
```

### Prompt Used

```text
Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: A classic-style sailboat with a dark hull and wooden mast is moored in a calm estuary during low tide. The water has receded, exposing a vast expanse of green, algae-covered mudflats behind the vessel. The boat, adorned with a string of small flags, floats peacefully, waiting for the tide to rise again.
- Capture metadata: Taken on 2026-05-02 18:33:45 BST (at 18:33:45 local time). GPS: 52.089294°N, 1.317741°E.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-05-04 21:20:44 BST by [check_models](https://github.com/jrp2014/check_models)._
