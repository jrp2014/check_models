# Diagnostics Report — 2 failure(s), 7 harness issue(s) (mlx-vlm 0.4.5)

## Summary

Automated benchmarking of **53 locally-cached VLM models** found **2 hard
failure(s)** and **7 harness/integration issue(s)** plus **0 preflight
compatibility warning(s)** in successful models. 51 of 53 models succeeded.

Test image: `20260502-152439_DSC09840.jpg` (17.3 MB).

---

## Issue Queue

Root-cause issue drafts are generated in [issues/index.md](../issues/index.md)
and grouped by owner, subtype, and normalized symptom family.

| Owner                            | Issue Subtype                           |   Affected Model Count | Representative Model                                    | Issue Draft                                                                                                                                                                   | Acceptance Signal                                                                                                                              |
|----------------------------------|-----------------------------------------|------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx`                            | `MLX_MODEL_LOAD_MODEL`                  |                      1 | `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | [`mlx_mlx-model-load-model_001`](../issues/issue_001_mlx_mlx-model-load-model_001.md)                                                                                         | Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer. |
| `model configuration/repository` | `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR` |                      1 | `mlx-community/MolmoPoint-8B-fp16`                      | [`model-configuration-repository_model-config-processor-load-processor_001`](../issues/issue_002_model-configuration-repository_model-config-processor-load-processor_001.md) | Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer. |
| `mlx-vlm`                        | `encoding`                              |                      1 | `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [`mlx-vlm_encoding_001`](../issues/issue_003_mlx-vlm_encoding_001.md)                                                                                                         | Affected reruns contain no leaked BPE, byte-level, or tokenizer marker text.                                                                   |
| `mlx-vlm`                        | `stop_token`                            |                      2 | `microsoft/Phi-3.5-vision-instruct`                     | [`mlx-vlm_stop-token_001`](../issues/issue_004_mlx-vlm_stop-token_001.md)                                                                                                     | Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete. |
| `model-config / mlx-vlm`         | `prompt_template`                       |                      2 | `mlx-community/llava-v1.6-mistral-7b-8bit`              | [`model-config-mlx-vlm_prompt-template_001`](../issues/issue_005_model-config-mlx-vlm_prompt-template_001.md)                                                                 | Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.          |
| `model-config / mlx-vlm`         | `stop_token`                            |                      1 | `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [`model-config-mlx-vlm_stop-token_001`](../issues/issue_006_model-config-mlx-vlm_stop-token_001.md)                                                                           | Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete. |
| `mlx-vlm / mlx`                  | `long_context`                          |                      1 | `mlx-community/X-Reasoner-7B-8bit`                      | [`mlx-vlm-mlx_long-context_001`](../issues/issue_007_mlx-vlm-mlx_long-context_001.md)                                                                                         | A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.               |
| `mlx-vlm / mlx`                  | `long_context`                          |                      1 | `mlx-community/Qwen3.5-27B-4bit`                        | [`mlx-vlm-mlx_long-context_002`](../issues/issue_008_mlx-vlm-mlx_long-context_002.md)                                                                                         | A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.               |

---

## Action Summary

Owner-first triage with subtype, affected count, and next action.

### 1. mlx

- _Owner:_ `mlx`
- _Subtype:_ MLX_MODEL_LOAD_MODEL
- _Issue:_ Model loading failed: Received 4 parameters not in model:
- _Affected:_ 1 model(s)
- _Next step:_ check tensor/cache behavior and memory pressure handling.

### 2. model configuration/repository

- _Owner:_ `model configuration/repository`
- _Subtype:_ MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- _Issue:_ Loaded processor has no image_processor; expected multimodal
  processor.
- _Affected:_ 1 model(s)
- _Next step:_ verify model config, tokenizer files, and revision alignment.

### 3. mlx-vlm

- _Owner:_ `mlx-vlm`
- _Subtype:_ harness/integration
- _Issue:_ Harness/integration warnings
- _Affected:_ 3 model(s)
- _Next step:_ check processor/chat-template wiring and generation kwargs.

### 4. mlx-vlm / mlx

- _Owner:_ `mlx-vlm / mlx`
- _Subtype:_ harness/integration
- _Issue:_ Harness/integration warnings
- _Affected:_ 1 model(s)
- _Next step:_ validate long-context handling and stop-token behavior across
  mlx-vlm + mlx runtime.

### 5. model-config / mlx-vlm

- _Owner:_ `model-config / mlx-vlm`
- _Subtype:_ harness/integration
- _Issue:_ Harness/integration warnings
- _Affected:_ 3 model(s)
- _Next step:_ validate chat-template/config expectations and mlx-vlm prompt
  formatting for this model.

### 6. mlx-vlm / mlx

- _Owner:_ `mlx-vlm / mlx`
- _Subtype:_ stack-signal
- _Issue:_ Stack-signal anomalies
- _Affected:_ 1 successful model(s)
- _Next step:_ validate long-context handling and stop-token behavior across
  mlx-vlm + mlx runtime.

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

[01:55:17] ERROR    Failed to load model mlx-community/Kimi-VL-A3B-Thinking-8bit
                    ╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16621 in _run_model_generation      │
                    │                                                                                                  │
                    │   16618 │   _set_failure_phase(phase_callback, "import")                                         │
                    │   16619 │   _ensure_generation_runtime_symbols()                                                 │
                    │   16620 │                                                                                        │
                    │ ❱ 16621 │   # Load model from HuggingFace Hub - this handles automatic download/caching          │
                    │   16622 │   # and converts weights to MLX format for Apple Silicon optimization                  │
                    │   16623 │   _set_failure_phase(phase_callback, "model_load")                                     │
                    │   16624 │   try:                                                                                 │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16034 in _load_model                │
                    │                                                                                                  │
                    │   16031 │   """Load model from HuggingFace Hub or local path.                                    │
                    │   16032 │                                                                                        │
                    │   16033 │   Args:                                                                                │
                    │ ❱ 16034 │   │   params: The parameters for image processing, including model identifier.         │
                    │   16035 │                                                                                        │
                    │   16036 │   Returns:                                                                             │
                    │   16037 │   │   Tuple of ``(model, processor, config)`` where ``processor`` is an                │
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
[01:55:18] DEBUG    HF Cache Info for mlx-community/Kimi-VL-A3B-Thinking-8bit: size=17023.6 MB, files=18
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
[01:56:50] ERROR    Model preflight validation failed for mlx-community/MolmoPoint-8B-fp16
                    ╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16392 in _prepare_generation_prompt │
                    │                                                                                                  │
                    │   16389 │   phase_timer: PhaseTimer | None,                                                      │
                    │   16390 ) -> str | list[object]:                                                                 │
                    │   16391 │   """Run preflight checks and build the prompt payload for generation."""              │
                    │ ❱ 16392 │   try:                                                                                 │
                    │   16393 │   │   chat_template_kwargs: ChatTemplateKwargs = (                                     │
                    │   16394 │   │   │   {"enable_thinking": True} if params.enable_thinking else {}                  │
                    │   16395 │   │   )                                                                                │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16184 in                            │
                    │ _run_model_preflight_validators                                                                  │
                    │                                                                                                  │
                    │   16181 │   │   )                                                                                │
                    │   16182 │                                                                                        │
                    │   16183 │   _set_failure_phase(phase_callback, "processor_load")                                 │
                    │ ❱ 16184 │   if not callable(processor):                                                          │
                    │   16185 │   │   _raise_preflight_error(                                                          │
                    │   16186 │   │   │   "Loaded processor is not callable.",                                         │
                    │   16187 │   │   │   phase="processor_load",                                                      │
                    │                                                                                                  │
                    │ /Users/jrp/Documents/AI/mlx/check_models/src/check_models.py:16117 in _raise_preflight_error     │
                    │                                                                                                  │
                    │   16114 │   │   return None                                                                      │
                    │   16115 │   if isinstance(config, Mapping):                                                      │
                    │   16116 │   │   config_map = cast("Mapping[str, object]", config)                                │
                    │ ❱ 16117 │   │   return config_map.get(key)                                                       │
                    │   16118 │   return getattr(config, key, None)                                                    │
                    │   16119                                                                                          │
                    │   16120                                                                                          │
                    ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
                    ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

## Harness/Integration Issues (7 model(s))

7 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `microsoft/Phi-3.5-vision-instruct`

- _Observed:_ Generation appears to continue through stop/control tokens
  instead of ending cleanly.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=1,394, output=500, output/prompt=35.87%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;\|end\|&gt; appeared in generated text.
  \| Special control token &lt;\|endoftext\|&gt; appeared in generated text.
  \| hit token cap (500) \| nontext prompt burden=65%
- _Evidence:_ Special control token &lt;\|end\|&gt; appeared in generated
  text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated
  text.
- _Token context:_ prompt=1,394 \| output/prompt=35.87% \| nontext burden=65%
  \| stop=completed \| hit token cap (500)
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Generated text appears to continue into example-code templates mid-output.
- Output switched language/script unexpectedly (tokenizer_artifact, code_snippet).

**Sample output:**

```text
Title: Woodbridge Tide Mill Museum and Boat

Description: The Woodbridge Tide Mill Museum in Woodbridge, Suffolk, England, is seen at low tide on the River Deben. In the foreground, a long boat covere...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Observed:_ Decoded output contains tokenizer artifacts that should not
  appear in user-facing text.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=2,682, output=124, output/prompt=4.62%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| encoding
- _Summary:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 78 occurrences). \| nontext prompt burden=82% \| missing
  sections: description, keywords \| missing terms: Bench, Blue sky, East
  Anglia, English countryside, Moored
- _Evidence:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 78 occurrences).
- _Token context:_ prompt=2,682 \| output/prompt=4.62% \| nontext burden=82%
  \| stop=completed
- _Next action:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 78 occurrences).
- Output omitted required Title/Description/Keywords sections (description, keywords).

**Sample output:**

```text
Title:ĠTideĠMillĠMuseumĠatĠLowĠTideĊĊDescription:ĠAĠlongĠboatĠcoveredĠwithĠaĠblackĠtarpĠrestsĠonĠtheĠmuddyĠriverbedĠinĠfrontĠofĠtheĠWoodbridgeĠTideĠMillĠMuseum.ĠTheĠhistoricĠwhiteĠbuildingĠstandsĠwith...
```

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Observed:_ Generation appears to continue through stop/control tokens
  instead of ending cleanly.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=6,681, output=500, output/prompt=7.48%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;/think&gt; appeared in generated text.
  \| hit token cap (500) \| nontext prompt burden=93% \| missing sections:
  title, keywords
- _Evidence:_ Special control token &lt;/think&gt; appeared in generated text.
- _Token context:_ prompt=6,681 \| output/prompt=7.48% \| nontext burden=93%
  \| stop=completed \| hit token cap (500)
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: <think>.
- Output omitted required Title/Description/Keywords sections (title, keywords).
- Output leaked reasoning or prompt-template text (<think>).

**Sample output:**

```text
<think>Got it, let's tackle this. First, I need to analyze the image for cataloguing metadata. The user provided a lot of context, but I need to focus on what's clearly visible.

First, the Title. It ...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Observed:_ Generation appears to continue through stop/control tokens
  instead of ending cleanly.
- _Likely owner:_ `model-config / mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate chat-template/config expectations and
  mlx-vlm prompt formatting for this model.
- _Token summary:_ prompt=16,794, output=2, output/prompt=0.01%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;\|endoftext\|&gt; appeared in generated
  text. \| Output appears truncated to about 2 tokens. \| nontext prompt
  burden=97% \| missing terms: Bench, Blue sky, Clouds, East Anglia, English
  countryside
- _Evidence:_ Special control token &lt;\|endoftext\|&gt; appeared in
  generated text. \| Output appears truncated to about 2 tokens.
- _Token context:_ prompt=16,794 \| output/prompt=0.01% \| nontext burden=97%
  \| stop=completed
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output appears truncated to about 2 tokens.
- At long prompt length (16794 tokens), output stayed unusually short (2 tokens; ratio 0.0%).
- Model output may not follow prompt or image contents (missing: Bench, Blue sky, Clouds, East Anglia, English countryside).
- Output switched language/script unexpectedly (tokenizer_artifact).

**Sample output:**

```text
<|endoftext|>
```

### `mlx-community/X-Reasoner-7B-8bit`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=16,794, output=500, output/prompt=2.98%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ cutoff_degraded \| long_context
- _Summary:_ At long prompt length (16794 tokens), output became repetitive.
  \| hit token cap (500) \| nontext prompt burden=97% \| missing terms: Bench,
  English countryside, Moored, Mudflats, Objects
- _Evidence:_ At long prompt length (16794 tokens), output became repetitive.
- _Token context:_ prompt=16,794 \| output/prompt=2.98% \| nontext burden=97%
  \| stop=completed \| hit token cap (500)
- _Next action:_ Inspect long-context cache behavior under heavy image-token
  burden.

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16794 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: phrase: "solar energy in museums,...").

**Sample output:**

```text
Title:
Woodbridge Tide Mill Museum at low tide

Description:
A long boat covered with a black tarp lies on the muddy riverbed at low tide, with a historic white building labeled "Tide Mill Museum" in ...
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Observed:_ Output shape suggests a prompt-template or stop-condition
  mismatch.
- _Likely owner:_ `model-config / mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate chat-template/config expectations and
  mlx-vlm prompt formatting for this model.
- _Token summary:_ prompt=0, output=0, output/prompt=n/a

**Maintainer triage:**

- _Likely owner:_ model-config \| confidence=high
- _Classification:_ harness \| prompt_template
- _Summary:_ Model returned zero output tokens.
- _Evidence:_ Model returned zero output tokens.
- _Token context:_ prompt=0 \| stop=completed
- _Next action:_ Inspect model repo config, chat template, and EOS settings.

**Why this appears to be an integration/runtime issue:**

- Model returned zero output tokens.

**Sample output:**

```text
<empty output>
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Observed:_ Output shape suggests a prompt-template or stop-condition
  mismatch.
- _Likely owner:_ `model-config / mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate chat-template/config expectations and
  mlx-vlm prompt formatting for this model.
- _Token summary:_ prompt=1,587, output=14, output/prompt=0.88%

**Maintainer triage:**

- _Likely owner:_ model-config \| confidence=high
- _Classification:_ harness \| prompt_template
- _Summary:_ Output is very short relative to prompt size (0.9%), suggesting
  possible early-stop or prompt-handling issues. \| nontext prompt burden=70%
  \| missing sections: title, description, keywords \| missing terms: Bench,
  Blue sky, Clouds, East Anglia, English countryside
- _Evidence:_ Output is very short relative to prompt size (0.9%), suggesting
  possible early-stop or prompt-handling issues.
- _Token context:_ prompt=1,587 \| output/prompt=0.88% \| nontext burden=70%
  \| stop=completed
- _Next action:_ Check chat-template and EOS defaults first; the output shape
  is not matching the requested contract.

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.9%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: Bench, Blue sky, Clouds, East Anglia, English countryside).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
- Use only the metadata that is clearly supported by the image.
```

---

### Long-Context Degradation / Potential Stack Issues (1 model(s))

1 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely
integration/runtime issues worth checking upstream.

| Model                            |   Prompt Tok |   Output Tok | Output/Prompt   | Symptom                                                                            | Owner           |
|----------------------------------|--------------|--------------|-----------------|------------------------------------------------------------------------------------|-----------------|
| `mlx-community/Qwen3.5-27B-4bit` |       16,807 |          500 | 2.97%           | Output degeneration under long prompt length (incomplete_sentence: ends with 'in') | `mlx-vlm / mlx` |

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

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 10
- **Summary diagnostics models:** 43
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1404.77s (1404.77s)
- **Average runtime per model:** 26.51s (26.51s)
- **Dominant runtime phase:** decode dominated 50/53 measured model runs (91% of tracked runtime).
- **Phase totals:** model load=120.19s, prompt prep=0.16s, decode=1271.45s, cleanup=6.61s
- **Observed stop reasons:** completed=51, exception=2
- **Validation overhead:** 10.58s total (avg 0.20s across 53 model(s)).
- **First-token latency:** Avg 14.32s | Min 0.07s | Max 85.76s across 50 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (43 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Clean output (5 model(s))

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`
- `mlx-community/gemma-3-27b-it-qat-4bit`
- `mlx-community/gemma-4-26b-a4b-it-4bit`
- `mlx-community/gemma-4-31b-it-4bit`
- `mlx-community/pixtral-12b-bf16`

### Ran, but with quality warnings (38 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Output became repetitive, indicating possible generation instability (token: phrase: "wooden house, wooden house,...").
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'a').
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections (title, keywords).
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections (title, description).
- `mlx-community/FastVLM-0.5B-bf16`: Output omitted required Title/Description/Keywords sections (keywords).
- `mlx-community/GLM-4.6V-Flash-6bit`: Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'if').
- `mlx-community/GLM-4.6V-nvfp4`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Output formatting deviated from the requested structure. Details: Unknown tags: <end_of_utterance>.
- `mlx-community/InternVL3-14B-8bit`: Nonvisual metadata borrowing
- `mlx-community/InternVL3-8B-bf16`: Nonvisual metadata borrowing
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Output omitted required Title/Description/Keywords sections (title, description).
- `mlx-community/LFM2-VL-1.6B-8bit`: Title length violation (4 words; expected 5-10)
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Title length violation (30 words; expected 5-10)
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: ⚠️REVIEW:context_budget
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: ⚠️REVIEW:context_budget
- `mlx-community/Molmo-7B-D-0924-8bit`: Title length violation (43 words; expected 5-10)
- `mlx-community/Molmo-7B-D-0924-bf16`: Title length violation (43 words; expected 5-10)
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Output appears to copy prompt context verbatim (45% overlap).
- `mlx-community/Qwen3.5-27B-mxfp8`: Likely capped by max token budget
- `mlx-community/Qwen3.5-35B-A3B-4bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.5-35B-A3B-6bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.5-35B-A3B-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.5-9B-MLX-4bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.6-27B-mxfp8`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/SmolVLM-Instruct-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "wooden house, wooden house,...").
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Output appears to copy prompt context verbatim (55% overlap).
- `mlx-community/gemma-3-27b-it-qat-8bit`: Nonvisual metadata borrowing
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents (missing: Bench, Blue sky, Clouds, East Anglia, English countrys...
- `mlx-community/gemma-3n-E4B-it-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/gemma-4-31b-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "- the image is...").
- `mlx-community/nanoLLaVA-1.5-4bit`: Output omitted required Title/Description/Keywords sections (keywords).
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: Model output may not follow prompt or image contents (missing: Bench, Blue sky, Clouds, East Anglia, English countrys...
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "the bottom of the...").
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents (missing: Bench, Blue sky, Clouds, East Anglia, English countrys...
- `mlx-community/pixtral-12b-8bit`: ⚠️REVIEW:context_budget
- `qnguyen3/nanoLLaVA`: Output omitted required Title/Description/Keywords sections (keywords).

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260502+e8ebdebe |
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260502-152439_DSC09840.jpg --exclude mlx-community/Qwen3-VL-2B-Thinking-bf16 Qwen/Qwen3-VL-2B-Instruct --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260502-152439_DSC09840.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Kimi-VL-A3B-Thinking-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260502-152439_DSC09840.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
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
- Description hint: The Woodbridge Tide Mill Museum in Woodbridge, Suffolk, England, is seen at low tide on the River Deben. In the foreground, a long boat covered by a black tarp rests on the muddy riverbed. A few people are sitting on benches in front of the historic white building on a sunny day.
- Keyword hints: Adobe Stock, Any Vision, Bench, Blue sky, Clouds, East Anglia, England, English countryside, Europe, Locations, Mill, Moored, Mudflats, Museum, Objects, People, Quay, River Deben, Riverbed, Rope
- Capture metadata: Taken on 2026-05-02 16:24:39 BST (at 16:24:39 local time). GPS: 52.091500°N, 1.318500°E.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260502-152439_DSC09840.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-05-03 02:16:09 BST by [check_models](https://github.com/jrp2014/check_models)._
