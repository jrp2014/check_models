# Diagnostics Report — 3 failure(s), 13 harness issue(s) (mlx-vlm 0.5.0)

**Run summary:** 55 locally-cached VLM model(s) checked; 3 hard failure(s), 13 harness/integration issue(s), 0 preflight warning(s), 52 successful run(s).

Test image: `20260515-201714_DSC09998_DxO.jpg` (27.0 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                                                      | Affected Models                                                 | Issue Draft                                                                                                                              | Evidence Bundle                                                                                                                                                                                          | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                                                                          | 1: `mlx-community/Kimi-VL-A3B-Thinking-8bit`                    | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_006_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json)             | Load/generation completes or fails with a narrower owner. |
| `mlx`                                                    | Weight/config mismatch during model load              | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                                                      | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                          | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx_mlx-model-load-weight-mismatch_001.md)   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_007_mlx-community_LFM2.5-VL-1.6B-bf16_MLX_MODEL_LOAD_WEIGHT_MISMATCH_7574b1189.json)            | Load/generation completes or fails with a narrower owner. |
| `mlx-lm`                                                 | Missing module/import during model load               | Model Error \| phase model_load \| ModuleNotFoundError                                                                                                                                                                 | 1: `facebook/pe-av-large`                                       | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-lm_mlx-lm-model-load-model_001.md)       | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_001_facebook_pe-av-large_MLX_LM_MODEL_LOAD_MODEL_b253df301723.json)                             | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;\|endoftext\|&gt; \| prompt_tokens=16868, prompt/image context dropped \| prompt=16,868 \| output/prompt=2.96% \| nontext burden=98% \| stop=max_tokens \| hit token cap (500) | 1: `mlx-community/X-Reasoner-7B-8bit`                           | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_stop-token_001.md)                   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_016_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_stop_token_001.json)                               | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~4 \| prompt=484 \| output/prompt=0.83% \| nontext burden=14% \| stop=completed                                                                                                                        | 1: `mlx-community/FastVLM-0.5B-bf16`                            | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_model-config-mlx-vlm_prompt-template_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_003_mlx-community_FastVLM-0.5B-bf16_model_config_mlx_vlm_prompt_template_001.json)              | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | generated_tokens~2 \| prompt_tokens=3581, output_tokens=2, output/prompt=0.1% \| prompt=3,581 \| output/prompt=0.06% \| nontext burden=88% \| stop=completed \| 3 model cluster                                        | 3: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (+2) | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm-mlx_long-context_001.md)             | [3 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_002_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_mlx_long_context_001.json) | Full and reduced reruns avoid context collapse.           |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | prompt_tokens=16868, prompt/image context dropped \| prompt=16,868 \| output/prompt=2.96% \| nontext burden=98% \| stop=max_tokens \| hit token cap (500) \| 8 model cluster                                           | 8: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+7)               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_007_mlx-vlm-mlx_long-context_002.md)             | [8 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_008_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_002.json)               | Full and reduced reruns avoid context collapse.           |

---

## 1. Failure affecting 1 model

- _Observed:_ Model loading failed: Model type pe_audio_video not supported.
- _Likely owner:_ `mlx-lm`
- _Why it matters:_ This prevented a complete model response; stage `Model
  Error`; phase `model_load`; code `MLX_LM_MODEL_LOAD_MODEL`; type
  `ValueError`.
- _Suggested next step:_ verify tokenizer/runtime compatibility and
  quantization settings.
- _Affected models:_ `facebook/pe-av-large`

**Representative maintainer triage:**

- _Likely owner:_ mlx-lm \| confidence=high
- _Classification:_ runtime_failure \| MLX_LM_MODEL_LOAD_MODEL
- _Summary:_ No module named 'mlx_lm.models.pe_audio_video'
- _Evidence:_ stage=Model Error \| phase=model_load \|
  code=MLX_LM_MODEL_LOAD_MODEL \| type=builtins.ModuleNotFoundError
- _Token context:_ stop=exception
- _Next action:_ Inspect the import path and installed package version that
  owns the missing module before treating this as a model failure.

| Model                  | Observed Behavior                                              | First Seen Failing      | Recent Repro           |
|------------------------|----------------------------------------------------------------|-------------------------|------------------------|
| `facebook/pe-av-large` | Model loading failed: Model type pe_audio_video not supported. | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `facebook/pe-av-large`

## 2. Failure affecting 1 model

- _Observed:_ Model loading failed: Received 4 parameters not in model:
  multi_modal_projector.linear_1.biases,
  multi_modal_projector.linear_1.scales,
  multi_modal_projector.linear_2.biases,
  multi_modal_projector.linear_2.scales.
- _Likely owner:_ `mlx`
- _Why it matters:_ This prevented a complete model response; stage `Model
  Error`; phase `model_load`; code `MLX_MODEL_LOAD_MODEL`; type `ValueError`.
- _Suggested next step:_ check tensor/cache behavior and memory pressure
  handling.
- _Affected models:_ `mlx-community/Kimi-VL-A3B-Thinking-8bit`

**Representative maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ runtime_failure \| MLX_MODEL_LOAD_MODEL
- _Summary:_ Received 4 parameters not in model:
  multi_modal_projector.linear_1.biases,
  multi_modal_projector.linear_1.scales,
  multi_modal_projector.linear_2.biases,
  multi_modal_projector.linear_2.scales.
- _Evidence:_ stage=Model Error \| phase=model_load \|
  code=MLX_MODEL_LOAD_MODEL \| type=builtins.ValueError
- _Token context:_ stop=exception
- _Next action:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.

| Model                                     | Observed Behavior                                                                                                                                                                                                     | First Seen Failing      | Recent Repro           |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | Model loading failed: Received 4 parameters not in model: multi_modal_projector.linear_1.biases, multi_modal_projector.linear_1.scales, multi_modal_projector.linear_2.biases, multi_modal_projector.linear_2.scales. | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/Kimi-VL-A3B-Thinking-8bit`

## 3. Failure affecting 1 model

- _Observed:_ Model loading failed: Missing 2 parameters:
  multi_modal_projector.layer_norm.bias,
  multi_modal_projector.layer_norm.weight.
- _Likely owner:_ `mlx`
- _Why it matters:_ This prevented a complete model response; stage `Weight
  Mismatch`; phase `model_load`; code `MLX_MODEL_LOAD_WEIGHT_MISMATCH`; type
  `ValueError`.
- _Suggested next step:_ check tensor/cache behavior and memory pressure
  handling.
- _Affected models:_ `mlx-community/LFM2.5-VL-1.6B-bf16`

**Representative maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ runtime_failure \| MLX_MODEL_LOAD_WEIGHT_MISMATCH
- _Summary:_ Missing 2 parameters: multi_modal_projector.layer_norm.bias,
  multi_modal_projector.layer_norm.weight.
- _Evidence:_ stage=Weight Mismatch \| phase=model_load \|
  code=MLX_MODEL_LOAD_WEIGHT_MISMATCH \| type=builtins.ValueError
- _Token context:_ stop=exception
- _Next action:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.

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

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=3,581, output=2, output/prompt=0.06%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ context_budget \| long_context
- _Summary:_ Output appears truncated to about 2 tokens. \| At long prompt
  length (3581 tokens), output stayed unusually short (2 tokens; ratio 0.1%).
  \| output/prompt=0.06% \| nontext prompt burden=88%
- _Evidence:_ generated_tokens~2 \| prompt_tokens=3581, output_tokens=2,
  output/prompt=0.1%
- _Token context:_ prompt=3,581 \| output/prompt=0.06% \| nontext burden=88%
  \| stop=completed
- _Next action:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 88% and the output stays weak under that load.

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 2 tokens.
- At long prompt length (3581 tokens), output stayed unusually short (2 tokens; ratio 0.1%).
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).

**Sample output:**

```text
rien
```

### `mlx-community/FastVLM-0.5B-bf16`

- _Observed:_ Output shape suggests a prompt-template or stop-condition
  mismatch.
- _Likely owner:_ `model-config / mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate chat-template/config expectations and
  mlx-vlm prompt formatting for this model.
- _Token summary:_ prompt=484, output=4, output/prompt=0.83%

**Maintainer triage:**

- _Likely owner:_ model-config \| confidence=high
- _Classification:_ harness \| prompt_template
- _Summary:_ Output appears truncated to about 4 tokens. \| missing terms:
  Rochester, Castle, turns, Red, celebrate
- _Evidence:_ generated_tokens~4
- _Token context:_ prompt=484 \| output/prompt=0.83% \| nontext burden=14% \|
  stop=completed
- _Next action:_ Inspect model repo config, chat template, and EOS settings.

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 4 tokens.
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).

**Sample output:**

```text
theyGREE.
```

### `mlx-community/InternVL3-14B-8bit`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=3,031, output=2, output/prompt=0.07%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ context_budget \| long_context
- _Summary:_ Output appears truncated to about 2 tokens. \| At long prompt
  length (3031 tokens), output stayed unusually short (2 tokens; ratio 0.1%).
  \| output/prompt=0.07% \| nontext prompt burden=86%
- _Evidence:_ generated_tokens~2 \| prompt_tokens=3031, output_tokens=2,
  output/prompt=0.1%
- _Token context:_ prompt=3,031 \| output/prompt=0.07% \| nontext burden=86%
  \| stop=completed
- _Next action:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 2 tokens.
- At long prompt length (3031 tokens), output stayed unusually short (2 tokens; ratio 0.1%).
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).

**Sample output:**

```text
ertz
```

### `mlx-community/InternVL3-8B-bf16`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=3,031, output=3, output/prompt=0.10%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ context_budget \| long_context
- _Summary:_ Output appears truncated to about 3 tokens. \| At long prompt
  length (3031 tokens), output stayed unusually short (3 tokens; ratio 0.1%).
  \| output/prompt=0.10% \| nontext prompt burden=86%
- _Evidence:_ generated_tokens~3 \| prompt_tokens=3031, output_tokens=3,
  output/prompt=0.1%
- _Token context:_ prompt=3,031 \| output/prompt=0.10% \| nontext burden=86%
  \| stop=completed
- _Next action:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.
- At long prompt length (3031 tokens), output stayed unusually short (3 tokens; ratio 0.1%).
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).

**Sample output:**

```text
xposite
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=16,868, output=500, output/prompt=2.96%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ cutoff_degraded \| long_context
- _Summary:_ At long prompt length (16868 tokens), output may stop following
  prompt/image context. \| hit token cap (500) \| nontext prompt burden=98% \|
  missing sections: title, description, keywords
- _Evidence:_ prompt_tokens=16868, prompt/image context dropped
- _Token context:_ prompt=16,868 \| output/prompt=2.96% \| nontext burden=98%
  \| stop=max_tokens \| hit token cap (500)
- _Next action:_ Inspect long-context cache behavior under heavy image-token
  burden.

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16868 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
The Best. 1, 10, 1, 100, 10, 10, 10, 1. The screen and 有声, 4K, 4, 4, 12, 10. The screen. A picture of a small, I, 10. We can see. A, 10, 1. 1. 1. 1. 10. A. It is the 1, 10. The screen, 10. 1. 10, 10, ...
```

### `mlx-community/Qwen3.5-27B-4bit`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=16,882, output=500, output/prompt=2.96%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ cutoff_degraded \| long_context
- _Summary:_ At long prompt length (16882 tokens), output may stop following
  prompt/image context. \| hit token cap (500) \| nontext prompt burden=98% \|
  missing sections: title, description, keywords
- _Evidence:_ prompt_tokens=16882, prompt/image context dropped
- _Token context:_ prompt=16,882 \| output/prompt=2.96% \| nontext burden=98%
  \| stop=max_tokens \| hit token cap (500)
- _Next action:_ Inspect long-context cache behavior under heavy image-token
  burden.

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16882 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
ండిuestr……j- Y- M- r- t- s- ( 1- 2- 3- 4- 5- 6- 7- 8- 9- 10- 11- 12- 13- 14- 15- 16- 17- 18- 19- 20- 21- 22- 23- 24- 25- 26- 27- 28- 29- 30- 31- 32- 33- 34- 35- 36- 37- 38- 39- 40- 41- 42- 43- 44- 45-...
```

### `mlx-community/Qwen3.5-27B-mxfp8`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=16,882, output=500, output/prompt=2.96%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ cutoff_degraded \| long_context
- _Summary:_ At long prompt length (16882 tokens), output became repetitive.
  \| hit token cap (500) \| nontext prompt burden=98% \| missing sections:
  title, description, keywords
- _Evidence:_ prompt_tokens=16882, repetitive output
- _Token context:_ prompt=16,882 \| output/prompt=2.96% \| nontext burden=98%
  \| stop=max_tokens \| hit token cap (500)
- _Next action:_ Inspect long-context cache behavior under heavy image-token
  burden.

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16882 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).
- Output became repetitive, indicating possible generation instability (token: 觉).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
evin满尼
憬觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
觉
...
```

### `mlx-community/Qwen3.5-35B-A3B-4bit`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=16,882, output=500, output/prompt=2.96%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ cutoff_degraded \| long_context
- _Summary:_ At long prompt length (16882 tokens), output became repetitive.
  \| hit token cap (500) \| nontext prompt burden=98% \| missing sections:
  title, description, keywords
- _Evidence:_ prompt_tokens=16882, repetitive output
- _Token context:_ prompt=16,882 \| output/prompt=2.96% \| nontext burden=98%
  \| stop=max_tokens \| hit token cap (500)
- _Next action:_ Inspect long-context cache behavior under heavy image-token
  burden.

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16882 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).
- Output became repetitive, indicating possible generation instability (token: `).
- Output contains corrupted or malformed text segments (character_loop: ' `' repeated).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
erval ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ...
```

### `mlx-community/Qwen3.5-35B-A3B-6bit`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=16,882, output=500, output/prompt=2.96%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ cutoff_degraded \| long_context
- _Summary:_ At long prompt length (16882 tokens), output became repetitive.
  \| hit token cap (500) \| nontext prompt burden=98% \| missing sections:
  title, description, keywords
- _Evidence:_ prompt_tokens=16882, repetitive output
- _Token context:_ prompt=16,882 \| output/prompt=2.96% \| nontext burden=98%
  \| stop=max_tokens \| hit token cap (500)
- _Next action:_ Inspect long-context cache behavior under heavy image-token
  burden.

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16882 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).
- Output became repetitive, indicating possible generation instability (token: 1).
- Output contains corrupted or malformed text segments (character_loop: '  1' repeated).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
ổng是分;    1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1...
```

### `mlx-community/Qwen3.5-35B-A3B-bf16`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=16,882, output=500, output/prompt=2.96%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ cutoff_degraded \| long_context
- _Summary:_ At long prompt length (16882 tokens), output became repetitive.
  \| hit token cap (500) \| nontext prompt burden=98% \| missing sections:
  title, description, keywords
- _Evidence:_ prompt_tokens=16882, repetitive output
- _Token context:_ prompt=16,882 \| output/prompt=2.96% \| nontext burden=98%
  \| stop=max_tokens \| hit token cap (500)
- _Next action:_ Inspect long-context cache behavior under heavy image-token
  burden.

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16882 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).
- Output became repetitive, indicating possible generation instability (token: was).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
fs, being made was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was was w...
```

### `mlx-community/Qwen3.5-9B-MLX-4bit`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=16,882, output=500, output/prompt=2.96%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ cutoff_degraded \| long_context
- _Summary:_ At long prompt length (16882 tokens), output became repetitive.
  \| hit token cap (500) \| nontext prompt burden=98% \| missing sections:
  title, description, keywords
- _Evidence:_ prompt_tokens=16882, repetitive output
- _Token context:_ prompt=16,882 \| output/prompt=2.96% \| nontext burden=98%
  \| stop=max_tokens \| hit token cap (500)
- _Next action:_ Inspect long-context cache behavior under heavy image-token
  burden.

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16882 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).
- Output became repetitive, indicating possible generation instability (token: phrase: "2 日 2 日...").
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
厄afaaktor捷2月沧海一imd,我wi,我wi,年，请wi,我wi,年

  90- 1. 2024 年 2 月  2 月  2 日  2 日  2 日  2 日  2 2024 年 2 月 2 日 2 日 2 日 2 日 2 日 2 日 2024 年 2 日 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2 日 2 日 2024 年 2 日 2 日 2024 ...
```

### `mlx-community/Qwen3.6-27B-mxfp8`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=16,882, output=500, output/prompt=2.96%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ cutoff_degraded \| long_context
- _Summary:_ At long prompt length (16882 tokens), output became repetitive.
  \| hit token cap (500) \| nontext prompt burden=98% \| missing sections:
  title, description, keywords
- _Evidence:_ prompt_tokens=16882, repetitive output
- _Token context:_ prompt=16,882 \| output/prompt=2.96% \| nontext burden=98%
  \| stop=max_tokens \| hit token cap (500)
- _Next action:_ Inspect long-context cache behavior under heavy image-token
  burden.

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16882 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).
- Output became repetitive, indicating possible generation instability (token: phrase: "3 \, y, y...").
- Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'y').
- Output formatting deviated from the requested structure. Details: Unknown tags: <EOF>.
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
弄清 0 >-< <EOF>
 3   \, y, y
\ 3   \, y, y
\ 3   \, y, y
\ 3   \, y, y
\ 3   \, y, y
\ 3   \, y, y
\ 3   \, y, y
\ 3   \, y, y
\ 3   \, y, y
\ 3   \, y, y
\ 3   \, y, y
\ 3   \, y, y
\ 3   \, y, y
\ 3 ...
```

### `mlx-community/X-Reasoner-7B-8bit`

- _Observed:_ Generation appears to continue through stop/control tokens
  instead of ending cleanly.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=16,868, output=500, output/prompt=2.96%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;\|endoftext\|&gt; appeared in generated
  text. \| At long prompt length (16868 tokens), output may stop following
  prompt/image context. \| hit token cap (500) \| nontext prompt burden=98%
- _Evidence:_ decoded text contains control token &lt;\|endoftext\|&gt; \|
  prompt_tokens=16868, prompt/image context dropped
- _Token context:_ prompt=16,868 \| output/prompt=2.96% \| nontext burden=98%
  \| stop=max_tokens \| hit token cap (500)
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|endoftext|&gt; appeared in generated text.
- At long prompt length (16868 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).
- Output switched language/script unexpectedly (tokenizer_artifact).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
<|endoftext|>1.<|endoftext|>1<|endoftext|>1<|endoftext|>10<|endoftext|>10<|endoftext|>10<|endoftext|>10<|endoftext|>10<|endoftext|>10.<|endoftext|>10.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|...
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none

| Model                                     | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------------|--------------------------|-------------------------|------------------------|
| `facebook/pe-av-large`                    | still failing            | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/LFM2.5-VL-1.6B-bf16`       | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 16
- **Summary diagnostics models:** 39
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1703.54s (1703.54s)
- **Average runtime per model:** 30.97s (30.97s)
- **Dominant runtime phase:** post-prefill decode dominated 36/55 measured model runs (52% of tracked runtime).
- **Phase totals:** model load=117.36s, local prompt prep=0.18s, upstream prefill / first-token=697.13s, post-prefill decode=873.98s, cleanup=6.02s
- **Generation total:** 1571.11s across 52 model(s); upstream prefill / first-token split available for 52/52 model(s).
- **Observed stop reasons:** completed=13, exception=3, max_tokens=39
- **Validation overhead:** 12.03s total (avg 0.22s across 55 model(s)).
- **Upstream prefill / first-token latency:** Avg 13.41s | Min 0.04s | Max 104.42s across 52 model(s).
- **What this likely means:** Most measured runtime is spent generating after the first token is available.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (39 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Ran, but with quality warnings:** 39 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260515-201714_DSC09998_DxO.jpg --exclude mlx-community/Qwen3-VL-2B-Thinking-bf16 Qwen/Qwen3-VL-2B-Instruct --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `facebook/pe-av-large`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/InternVL3-14B-8bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3.5-27B-4bit`, `mlx-community/Qwen3.5-27B-mxfp8`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-35B-A3B-6bit`, `mlx-community/Qwen3.5-35B-A3B-bf16`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/X-Reasoner-7B-8bit`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260515-201714_DSC09998_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-05-16 00:16:45 BST by [check_models](https://github.com/jrp2014/check_models)._

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260515+7b7c1240 |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.8.0.dev0                  |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.15.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.5.0               |
| macOS Version   | 26.5                        |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |

