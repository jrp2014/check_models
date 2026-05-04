# Diagnostics Report — 3 failure(s), 2 harness issue(s) (mlx-vlm 0.4.5)

**Run summary:** 54 locally-cached VLM model(s) checked; 3 hard failure(s), 2 harness/integration issue(s), 0 preflight warning(s), 51 successful run(s).

Test image: `20260502-173345_DSC09912_DxO.jpg` (27.3 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](../issues/index.md). Each row is intended to become one
focused upstream GitHub issue.

| Target                           | Problem                                                                                                                | Affected Models                                            | Issue Draft                                                                                                    | Evidence Bundle                                                                                                                     | Fixed When                                                |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                            | MLX: Model load / model error: Received 4 parameters not in model:                                                     | 1: `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | [issue draft](../issues/issue_001_mlx_mlx-model-load-model_001.md)                                             | [repro JSON](../repro_bundles/20260503T224052Z_002_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json)  | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                        | mlx-vlm: Model load / model error: Model type granite not supported. Error: No module named 'mlx_vlm.spe               | 1: `mlx-community/granite-4.1-8b-mxfp8`                    | [issue draft](../issues/issue_002_mlx-vlm_mlx-vlm-model-load-model_001.md)                                     | [repro JSON](../repro_bundles/20260503T224052Z_005_mlx-community_granite-4.1-8b-mxfp8_MLX_VLM_MODEL_LOAD_MODEL_5af3e849109f.json)   | Load/generation completes or fails with a narrower owner. |
| `model configuration/repository` | Model config: Processor load / processor error: Loaded processor has no image_processor; expected multimodal processor | 1: `mlx-community/MolmoPoint-8B-fp16`                      | [issue draft](../issues/issue_003_model-configuration-repository_model-config-processor-load-processor_001.md) | [repro JSON](../repro_bundles/20260503T224052Z_003_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json)  | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                        | Tokenizer / decoding artifact: Tokenizer space-marker artifacts (for example  ) appeared in output (                   | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](../issues/issue_004_mlx-vlm_encoding_001.md)                                                     | [repro JSON](../repro_bundles/20260503T224052Z_001_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json) | No BPE/byte markers in output.                            |
| `mlx-vlm / mlx`                  | Long-context collapse: At long prompt length (16901 tokens), output became repetitive                                  | 1: `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [issue draft](../issues/issue_005_mlx-vlm-mlx_long-context_001.md)                                             | [repro JSON](../repro_bundles/20260503T224052Z_004_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)       | Full and reduced reruns avoid context collapse.           |

---

## Upstream Filing Notes

File one upstream issue per row above. The linked issue drafts are the
pasteable bodies; this diagnostics file is the run-level queue and appendix.

- **Issue drafts:** 5 root-cause cluster(s).
- **Suggested targets:** `mlx`=1, `mlx-vlm`=2, `mlx-vlm / mlx`=1, `model configuration/repository`=1.
- **Standalone evidence:** each issue draft includes minimal inline evidence plus an exact cluster rerun command before any appendix detail.
- **Supporting files:** `repro JSON` links point to local repro bundles with prompt, environment, and generated-output context.

## Appendix

The remaining sections keep full run evidence for audit/debugging. They are not intended to be pasted wholesale into an upstream issue.

---

## 1. Failure affecting 1 model

- _Observed:_ Model loading failed: Received 4 parameters not in model:
- _Likely owner:_ `mlx`
- _Why it matters:_ This prevented a complete model response; stage `Model
  Error`; phase `model_load`; code `MLX_MODEL_LOAD_MODEL`; type `ValueError`.
- _Suggested next step:_ check tensor/cache behavior and memory pressure
  handling.
- _Affected models:_ `mlx-community/Kimi-VL-A3B-Thinking-8bit`

**Representative maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ runtime_failure \| MLX_MODEL_LOAD_MODEL
- _Summary:_ model error \| mlx model load model
- _Evidence:_ model error \| mlx model load model
- _Token context:_ stop=exception
- _Next action:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.

| Model                                     | Observed Behavior                                         | First Seen Failing      | Recent Repro           |
|-------------------------------------------|-----------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | Model loading failed: Received 4 parameters not in model: | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `mlx-community/Kimi-VL-A3B-Thinking-8bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

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

[23:24:54] ERROR    Failed to load model mlx-community/Kimi-VL-A3B-Thinking-8bit
                    ╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16703 in _run_model_generation      │
                    │                                                                                                  │
                    │   16700 │   try:                                                                                 │
                    │   16701 │   │   if phase_timer is not None:                                                      │
                    │   16702 │   │   │   with phase_timer.track("model_load"):                                        │
                    │ ❱ 16703 │   │   │   │   model, processor, config = _load_model(params)                           │
                    │   16704 │   │   else:                                                                            │
                    │   16705 │   │   │   model, processor, config = _load_model(params)                               │
                    │   16706 │   except Exception as load_err:                                                        │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16116 in _load_model                │
                    │                                                                                                  │
                    │   16113 │   │   Tuple of ``(model, processor, config)`` where ``processor`` is an                │
                    │   16114 │   │   ``transformers.ProcessorMixin`` and ``config`` may be ``None``.                  │
                    │   16115 │   """                                                                                  │
                    │ ❱ 16116 │   model, processor = load(                                                             │
                    │   16117 │   │   path_or_hf_repo=params.model_identifier,                                         │
                    │   16118 │   │   adapter_path=params.adapter_path,                                                │
                    │   16119 │   │   lazy=params.lazy,                                                                │
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
[23:24:54] DEBUG    HF Cache Info for mlx-community/Kimi-VL-A3B-Thinking-8bit: size=17023.6 MB, files=18
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
- _Next action:_ Inspect model repo config, chat template, and EOS settings.

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
[23:26:16] ERROR    Model preflight validation failed for mlx-community/MolmoPoint-8B-fp16
                    ╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16474 in _prepare_generation_prompt │
                    │                                                                                                  │
                    │   16471 │   │   )                                                                                │
                    │   16472 │   │   if phase_timer is not None:                                                      │
                    │   16473 │   │   │   with phase_timer.track("prompt_prep"):                                       │
                    │ ❱ 16474 │   │   │   │   _run_model_preflight_validators(                                         │
                    │   16475 │   │   │   │   │   model_identifier=params.model_identifier,                            │
                    │   16476 │   │   │   │   │   processor=processor,                                                 │
                    │   16477 │   │   │   │   │   config=config,                                                       │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16266 in                            │
                    │ _run_model_preflight_validators                                                                  │
                    │                                                                                                  │
                    │   16263 │   │   │   phase="processor_load",                                                      │
                    │   16264 │   │   )                                                                                │
                    │   16265 │   if getattr(processor, "image_processor", None) is None:                              │
                    │ ❱ 16266 │   │   _raise_preflight_error(                                                          │
                    │   16267 │   │   │   "Loaded processor has no image_processor; expected multimodal processor.",   │
                    │   16268 │   │   │   phase="processor_load",                                                      │
                    │   16269 │   │   )                                                                                │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16199 in _raise_preflight_error     │
                    │                                                                                                  │
                    │   16196                                                                                          │
                    │   16197 def _raise_preflight_error(message: str, *, phase: str) -> NoReturn:                     │
                    │   16198 │   """Raise a preflight ValueError annotated with the failing phase."""                 │
                    │ ❱ 16199 │   raise _tag_exception_failure_phase(ValueError(message), phase)                       │
                    │   16200                                                                                          │
                    │   16201                                                                                          │
                    │   16202 def _validate_model_artifact_layout(                                                     │
                    ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
                    ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

## 3. Failure affecting 1 model

- _Observed:_ Model loading failed: Model type granite not supported. Error:
  No module named 'mlx_vlm.speculative.drafters.granite'
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ This prevented a complete model response; stage `Model
  Error`; phase `model_load`; code `MLX_VLM_MODEL_LOAD_MODEL`; type
  `ValueError`.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Affected models:_ `mlx-community/granite-4.1-8b-mxfp8`

**Representative maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ runtime_failure \| MLX_VLM_MODEL_LOAD_MODEL
- _Summary:_ model error \| mlx vlm model load model
- _Evidence:_ model error \| mlx vlm model load model
- _Token context:_ stop=exception
- _Next action:_ Inspect prompt-template, stop-token, and decode
  post-processing behavior.

| Model                                | Observed Behavior                                                                                                     | First Seen Failing      | Recent Repro           |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/granite-4.1-8b-mxfp8` | Model loading failed: Model type granite not supported. Error: No module named 'mlx_vlm.speculative.drafters.granite' | 2026-05-03 23:40:52 BST | 1/1 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `mlx-community/granite-4.1-8b-mxfp8`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/granite-4.1-8b-mxfp8`

Traceback tail:

```text
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 136, in get_model_and_args
    raise ValueError(msg)
ValueError: Model type granite not supported. Error: No module named 'mlx_vlm.speculative.drafters.granite'
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: Model type granite not supported. Error: No module named 'mlx_vlm.speculative.drafters.granite'
```

Captured stdout/stderr:

```text
=== STDERR ===

[23:39:23] ERROR    Failed to load model mlx-community/granite-4.1-8b-mxfp8
                    ╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16703 in _run_model_generation      │
                    │                                                                                                  │
                    │   16700 │   try:                                                                                 │
                    │   16701 │   │   if phase_timer is not None:                                                      │
                    │   16702 │   │   │   with phase_timer.track("model_load"):                                        │
                    │ ❱ 16703 │   │   │   │   model, processor, config = _load_model(params)                           │
                    │   16704 │   │   else:                                                                            │
                    │   16705 │   │   │   model, processor, config = _load_model(params)                               │
                    │   16706 │   except Exception as load_err:                                                        │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16116 in _load_model                │
                    │                                                                                                  │
                    │   16113 │   │   Tuple of ``(model, processor, config)`` where ``processor`` is an                │
                    │   16114 │   │   ``transformers.ProcessorMixin`` and ``config`` may be ``None``.                  │
                    │   16115 │   """                                                                                  │
                    │ ❱ 16116 │   model, processor = load(                                                             │
                    │   16117 │   │   path_or_hf_repo=params.model_identifier,                                         │
                    │   16118 │   │   adapter_path=params.adapter_path,                                                │
                    │   16119 │   │   lazy=params.lazy,                                                                │
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
                    │ /Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py:236 in load_model                           │
                    │                                                                                                  │
                    │    233 │   with safetensors.safe_open(weight_files[0], framework="np") as f:                     │
                    │    234 │   │   is_mlx_format = f.metadata() and f.metadata().get("format") == "mlx"              │
                    │    235 │                                                                                         │
                    │ ❱  236 │   model_class, _ = get_model_and_args(config=config)                                    │
                    │    237 │                                                                                         │
                    │    238 │   # Initialize text and vision configs if not present                                   │
                    │    239 │   config.setdefault("text_config", config.pop("llm_config", {}))                        │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py:136 in get_model_and_args                   │
                    │                                                                                                  │
                    │    133 │                                                                                         │
                    │    134 │   msg = f"Model type {model_type} not supported. Error: {last_err}"                     │
                    │    135 │   logging.error(msg)                                                                    │
                    │ ❱  136 │   raise ValueError(msg)                                                                 │
                    │    137                                                                                           │
                    │    138                                                                                           │
                    │    139 def get_model_path(                                                                       │
                    ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
                    ValueError: Model type granite not supported. Error: No module named
                    'mlx_vlm.speculative.drafters.granite'
[23:39:24] DEBUG    HF Cache Info for mlx-community/granite-4.1-8b-mxfp8: size=8249.3 MB, files=10
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
- _Token summary:_ prompt=16,901, output=500, output/prompt=2.96%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ cutoff_degraded \| long_context
- _Summary:_ At long prompt length (16901 tokens), output became repetitive.
  \| hit token cap (500) \| nontext prompt burden=97% \| missing sections:
  title, description, keywords
- _Evidence:_ At long prompt length (16901 tokens), output became repetitive.
- _Token context:_ prompt=16,901 \| output/prompt=2.96% \| nontext burden=97%
  \| stop=completed \| hit token cap (500)
- _Next action:_ Inspect long-context cache behavior under heavy image-token
  burden.

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16901 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: classic, style, sailboat, dark, hull).
- Output became repetitive, indicating possible generation instability (token: phrase: "boat anchor boat anchor...").
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anc...
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none

| Model                                     | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------------|--------------------------|-------------------------|------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | still failing            | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16`        | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/granite-4.1-8b-mxfp8`      | new model failing        | 2026-05-03 23:40:52 BST | 1/1 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 5
- **Summary diagnostics models:** 49
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1214.71s (1214.71s)
- **Average runtime per model:** 22.49s (22.49s)
- **Dominant runtime phase:** decode dominated 50/54 measured model runs (90% of tracked runtime).
- **Phase totals:** model load=117.87s, prompt prep=0.16s, decode=1082.56s, cleanup=5.63s
- **Observed stop reasons:** completed=51, exception=3
- **Validation overhead:** 11.08s total (avg 0.21s across 54 model(s)).
- **First-token latency:** Avg 11.85s | Min 0.08s | Max 70.80s across 51 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

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
- `mlx-community/Molmo-7B-D-0924-bf16`: Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'to').
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
| mlx             | 0.32.0.dev20260503+e8ebdebe |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.7.0                       |
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

### Portable triage (no local image required)

```bash
# Capture dependency versions
python -m pip show mlx mlx-vlm mlx-lm transformers huggingface-hub tokenizers

# Verify imports with explicit pass/fail output
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
```

### Target specific failing models

**Note:** A comprehensive JSON reproduction bundle including system info and
the exact prompt trace has been exported to
[repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles)
for each failing model.

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Kimi-VL-A3B-Thinking-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
python -m check_models --image /Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/granite-4.1-8b-mxfp8
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

_Report generated on 2026-05-03 23:40:52 BST by [check_models](https://github.com/jrp2014/check_models)._
