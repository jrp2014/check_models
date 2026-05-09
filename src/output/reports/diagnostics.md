# Diagnostics Report — 7 failure(s), 5 harness issue(s) (mlx-vlm 0.5.0)

**Run summary:** 55 locally-cached VLM model(s) checked; 7 hard failure(s), 5 harness/integration issue(s), 0 preflight warning(s), 48 successful run(s).

Test image: `20260509-165009_DSC09954.jpg` (33.9 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

| Target                                                   | Problem                                                                                              | Evidence Snapshot                                                                                                                                                                                                        | Affected Models                                            | Issue Draft                                                                                                                                                              | Evidence Bundle                                                                                                                                                                                          | Fixed When                                                |
|----------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load                                                             | Model Error \| phase model_load \| ValueError \| 2 model cluster                                                                                                                                                         | 2: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (+1)                 | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)                                             | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T234107Z_001_LiquidAI_LFM2.5-VL-450M-MLX-bf16_MLX_MODEL_LOAD_MODEL_853049863f38.json)                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | mlx-vlm: Decode / runtime error: property 'text' of 'NaiveStreamingDetokenizer' object has no setter | Error \| phase decode \| AttributeError \| 3 model cluster                                                                                                                                                               | 3: `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (+2) | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_mlx-vlm-decode-error_001.md)                                         | [3 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T234107Z_005_mlx-community_ERNIE-4.5-VL-28B-A3B-Thinking-bf16_MLX_VLM_DECODE_ERROR_c6f291b6246e.json) | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Missing module/import during model load                                                              | Model Error \| phase model_load \| ValueError                                                                                                                                                                            | 1: `facebook/pe-av-large`                                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm_mlx-vlm-model-load-model_001.md)                                     | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T234107Z_002_facebook_pe-av-large_MLX_VLM_MODEL_LOAD_MODEL_8b244da8c605.json)                            | Load/generation completes or fails with a narrower owner. |
| model configuration / repository                         | Processor config is missing image processor                                                          | Processor Error \| phase processor_load \| ValueError                                                                                                                                                                    | 1: `mlx-community/MolmoPoint-8B-fp16`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_model-configuration-repository_model-config-processor-load-processor_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T234107Z_009_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json)             | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers                                                             | 57 BPE space markers found in decoded text \| prompt=2,676 \| output/prompt=3.48% \| nontext burden=82% \| stop=completed                                                                                                | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm_encoding_001.md)                                                     | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T234107Z_004_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json)            | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text                                                       | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=1,387 \| output/prompt=36.05% \| nontext burden=66% \| stop=max_tokens \| hit token cap (500) | 1: `microsoft/Phi-3.5-vision-instruct`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm_stop-token_001.md)                                                   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T234107Z_003_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)                              | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                                                                | output/prompt=0.4% \| prompt=2,789 \| output/prompt=0.39% \| nontext burden=83% \| stop=completed \| 2 model cluster                                                                                                     | 2: `mlx-community/llava-v1.6-mistral-7b-8bit` (+1)         | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_007_model-config-mlx-vlm_prompt-template_001.md)                                 | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T234107Z_012_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json)  | Requested sections render without template leakage.       |
| model repo first; mlx-vlm if template handling disagrees | Stop/control tokens leaked into generated text                                                       | decoded text contains control token &lt;\|endoftext\|&gt; \| generated_tokens~5 \| prompt=16,789 \| output/prompt=0.03% \| nontext burden=97% \| stop=completed                                                          | 1: `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_008_model-config-mlx-vlm_stop-token_001.md)                                      | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T234107Z_010_mlx-community_Qwen2-VL-2B-Instruct-4bit_model_config_mlx_vlm_stop_token_001.json)           | No leaked stop/control tokens.                            |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short                                                | token cap \| missing sections \| abrupt tail \| prompt=16,804 \| output/prompt=2.98% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500)                                                                     | 1: `mlx-community/Qwen3.5-9B-MLX-4bit`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_009_mlx-vlm-mlx_long-context_001.md)                                             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T234107Z_011_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_001.json)                        | Full and reduced reruns avoid context collapse.           |

---

## 1. Failure affecting 2 models

- _Observed:_ Model loading failed: Received 2 parameters not in model:
  multi_modal_projector.layer_norm.bias,
  multi_modal_projector.layer_norm.weight.
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
- _Summary:_ Received 2 parameters not in model:
  multi_modal_projector.layer_norm.bias,
  multi_modal_projector.layer_norm.weight.
- _Evidence:_ stage=Model Error \| phase=model_load \|
  code=MLX_MODEL_LOAD_MODEL \| type=builtins.ValueError
- _Token context:_ stop=exception
- _Next action:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.

| Model                                     | Observed Behavior                                                                                                                                                                                                     | First Seen Failing      | Recent Repro           |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`        | Model loading failed: Received 2 parameters not in model: multi_modal_projector.layer_norm.bias, multi_modal_projector.layer_norm.weight.                                                                             | 2026-05-04 20:21:24 BST | 3/3 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | Model loading failed: Received 4 parameters not in model: multi_modal_projector.linear_1.biases, multi_modal_projector.linear_1.scales, multi_modal_projector.linear_2.biases, multi_modal_projector.linear_2.scales. | 2026-02-13 13:18:47 GMT | 3/3 recent runs failed |

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

## 2. Failure affecting 1 model

- _Observed:_ Model loading failed: Model type pe_audio_video not supported.
  Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video'
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ This prevented a complete model response; stage `Model
  Error`; phase `model_load`; code `MLX_VLM_MODEL_LOAD_MODEL`; type
  `ValueError`.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Affected models:_ `facebook/pe-av-large`

**Representative maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ runtime_failure \| MLX_VLM_MODEL_LOAD_MODEL
- _Summary:_ Model type pe_audio_video not supported. Error: No module named
  'mlx_vlm.speculative.drafters.pe_audio_video'
- _Evidence:_ stage=Model Error \| phase=model_load \|
  code=MLX_VLM_MODEL_LOAD_MODEL \| type=builtins.ValueError
- _Token context:_ stop=exception
- _Next action:_ Inspect the import path and installed package version that
  owns the missing module before treating this as a model failure.

| Model                  | Observed Behavior                                                                                                                   | First Seen Failing      | Recent Repro           |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `facebook/pe-av-large` | Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video' | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `facebook/pe-av-large`

## 3. Failure affecting 1 model

- _Observed:_ Model runtime error during generation for
  mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16: property 'text' of
  'NaiveStreamingDetokenizer' object has no setter
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ This prevented a complete model response; stage `Error`;
  phase `decode`; code `MLX_VLM_DECODE_ERROR`; type `ValueError`.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Affected models:_ `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

**Representative maintainer triage:**

- _Likely owner:_ model \| confidence=high
- _Classification:_ model_shortcoming \| MLX_VLM_DECODE_ERROR
- _Summary:_ keywords=58 \| context echo=100% \| nonvisual metadata reused \|
  reasoning leak
- _Evidence:_ instruction echo \| metadata borrowing \| hallucination
- _Token context:_ stop=exception
- _Next action:_ Treat as a model-quality limitation for this prompt and
  image.

| Model                                              | Observed Behavior                                                                                                                                               | First Seen Failing      | Recent Repro           |
|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | Model runtime error during generation for mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16: property 'text' of 'NaiveStreamingDetokenizer' object has no setter | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

## 4. Failure affecting 1 model

- _Observed:_ Model runtime error during generation for
  mlx-community/LFM2-VL-1.6B-8bit: property 'text' of
  'NaiveStreamingDetokenizer' object has no setter
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ This prevented a complete model response; stage `Error`;
  phase `decode`; code `MLX_VLM_DECODE_ERROR`; type `ValueError`.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Affected models:_ `mlx-community/LFM2-VL-1.6B-8bit`

**Representative maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;\|im_end\|&gt; appeared in generated
  text. \| keywords=57 \| context echo=100% \| nonvisual metadata reused
- _Evidence:_ decoded text contains control token &lt;\|im_end\|&gt;
- _Token context:_ stop=exception
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

| Model                             | Observed Behavior                                                                                                                              | First Seen Failing      | Recent Repro           |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/LFM2-VL-1.6B-8bit` | Model runtime error during generation for mlx-community/LFM2-VL-1.6B-8bit: property 'text' of 'NaiveStreamingDetokenizer' object has no setter | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/LFM2-VL-1.6B-8bit`

## 5. Failure affecting 1 model

- _Observed:_ Model runtime error during generation for
  mlx-community/LFM2.5-VL-1.6B-bf16: property 'text' of
  'NaiveStreamingDetokenizer' object has no setter
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ This prevented a complete model response; stage `Error`;
  phase `decode`; code `MLX_VLM_DECODE_ERROR`; type `ValueError`.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Affected models:_ `mlx-community/LFM2.5-VL-1.6B-bf16`

**Representative maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;\|im_end\|&gt; appeared in generated
  text. \| keywords=57 \| context echo=100% \| nonvisual metadata reused
- _Evidence:_ decoded text contains control token &lt;\|im_end\|&gt;
- _Token context:_ stop=exception
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

| Model                               | Observed Behavior                                                                                                                                | First Seen Failing      | Recent Repro           |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16` | Model runtime error during generation for mlx-community/LFM2.5-VL-1.6B-bf16: property 'text' of 'NaiveStreamingDetokenizer' object has no setter | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/LFM2.5-VL-1.6B-bf16`

## 6. Failure affecting 1 model

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
- _Summary:_ Loaded processor has no image_processor; expected multimodal
  processor.
- _Evidence:_ stage=Processor Error \| phase=processor_load \|
  code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR \| type=builtins.ValueError
- _Token context:_ stop=exception
- _Next action:_ Inspect the model repo processor/preprocessor config and
  AutoProcessor mapping; the multimodal processor is missing or not exposing
  the image processor expected by mlx-vlm.

| Model                              | Observed Behavior                                                       | First Seen Failing      | Recent Repro           |
|------------------------------------|-------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/MolmoPoint-8B-fp16` | Loaded processor has no image_processor; expected multimodal processor. | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/MolmoPoint-8B-fp16`

---

## Harness/Integration Issues (5 model(s))

5 model(s) show potential harness/integration issues; see per-model breakdown
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
- _Token summary:_ prompt=1,387, output=500, output/prompt=36.05%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;\|end\|&gt; appeared in generated text.
  \| Special control token &lt;\|endoftext\|&gt; appeared in generated text.
  \| hit token cap (500) \| nontext prompt burden=66%
- _Evidence:_ decoded text contains control token &lt;\|end\|&gt; \| decoded
  text contains control token &lt;\|endoftext\|&gt;
- _Token context:_ prompt=1,387 \| output/prompt=36.05% \| nontext burden=66%
  \| stop=max_tokens \| hit token cap (500)
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

**Sample output:**

```text
Title: Gothic Revival Church in Petersfield, Hampshire

Description: A Gothic Revival style church with a tall spire and flint walls stands prominently against a bright blue sky with wispy clouds. A b...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Observed:_ Decoded output contains tokenizer artifacts that should not
  appear in user-facing text.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=2,676, output=93, output/prompt=3.48%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| encoding
- _Summary:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 57 occurrences). \| nontext prompt burden=82% \| missing
  sections: description, keywords \| missing terms: Dorking, Gothic
  Architecture, Objects, Surrey, low
- _Evidence:_ 57 BPE space markers found in decoded text
- _Token context:_ prompt=2,676 \| output/prompt=3.48% \| nontext burden=82%
  \| stop=completed
- _Next action:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 57 occurrences).
- Output omitted required Title/Description/Keywords sections (description, keywords).

**Sample output:**

```text
Title:ĠGothicĠRevivalĠChurchĠwithĠSpireĊĊDescription:ĠAĠGothicĠRevivalĠstyleĠchurchĠwithĠaĠtallĠspireĠandĠflintĠwallsĠstandsĠagainstĠaĠbrightĠblueĠskyĠwithĠwispyĠclouds.ĠAĠblackĠcarĠisĠparkedĠinĠtheĠf...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Observed:_ Generation appears to continue through stop/control tokens
  instead of ending cleanly.
- _Likely owner:_ `model-config / mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate chat-template/config expectations and
  mlx-vlm prompt formatting for this model.
- _Token summary:_ prompt=16,789, output=5, output/prompt=0.03%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;\|endoftext\|&gt; appeared in generated
  text. \| Output appears truncated to about 5 tokens. \| nontext prompt
  burden=97% \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church
- _Evidence:_ decoded text contains control token &lt;\|endoftext\|&gt; \|
  generated_tokens~5
- _Token context:_ prompt=16,789 \| output/prompt=0.03% \| nontext burden=97%
  \| stop=completed
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output appears truncated to about 5 tokens.
- At long prompt length (16789 tokens), output stayed unusually short (5 tokens; ratio 0.0%).
- Model output may not follow prompt or image contents (missing: Bell Tower, Blue sky, Car, Chapel, Church).
- Output switched language/script unexpectedly (tokenizer_artifact).

**Sample output:**

```text
<|endoftext|><|endoftext|><|endoftext|><|endoftext|>
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

- _Observed:_ Output shape suggests a prompt-template or stop-condition
  mismatch.
- _Likely owner:_ `model-config / mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate chat-template/config expectations and
  mlx-vlm prompt formatting for this model.
- _Token summary:_ prompt=2,789, output=11, output/prompt=0.39%

**Maintainer triage:**

- _Likely owner:_ model-config \| confidence=high
- _Classification:_ harness \| prompt_template
- _Summary:_ Output is very short relative to prompt size (0.4%), suggesting
  possible early-stop or prompt-handling issues. \| nontext prompt burden=83%
  \| missing sections: title, description, keywords \| missing terms: Bell
  Tower, Blue sky, Car, Chapel, Cross
- _Evidence:_ output/prompt=0.4%
- _Token context:_ prompt=2,789 \| output/prompt=0.39% \| nontext burden=83%
  \| stop=completed
- _Next action:_ Check chat-template and EOS defaults first; the output shape
  is not matching the requested contract.

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: Bell Tower, Blue sky, Car, Chapel, Cross).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
The image is a photograph of a church.
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

- _Observed:_ Output shape suggests a prompt-template or stop-condition
  mismatch.
- _Likely owner:_ `model-config / mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate chat-template/config expectations and
  mlx-vlm prompt formatting for this model.
- _Token summary:_ prompt=1,585, output=9, output/prompt=0.57%

**Maintainer triage:**

- _Likely owner:_ model-config \| confidence=high
- _Classification:_ harness \| prompt_template
- _Summary:_ Output is very short relative to prompt size (0.6%), suggesting
  possible early-stop or prompt-handling issues. \| nontext prompt burden=70%
  \| missing terms: Bell Tower, Blue sky, Car, Chapel, Church
- _Evidence:_ output/prompt=0.6%
- _Token context:_ prompt=1,585 \| output/prompt=0.57% \| nontext burden=70%
  \| stop=completed
- _Next action:_ Inspect model repo config, chat template, and EOS settings.

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: Bell Tower, Blue sky, Car, Chapel, Church).

**Sample output:**

```text
- Use only the above metadata hints.
```

---

### Long-Context Degradation / Potential Stack Issues (1 model(s))

1 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely
integration/runtime issues worth checking upstream.

| Model                               |   Prompt Tok |   Output Tok | Output/Prompt   | Symptom                                                                            | Owner           |
|-------------------------------------|--------------|--------------|-----------------|------------------------------------------------------------------------------------|-----------------|
| `mlx-community/Qwen3.5-9B-MLX-4bit` |       16,804 |          500 | 2.98%           | Output degeneration under long prompt length (incomplete_sentence: ends with 'fl') | `mlx-vlm / mlx` |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none

| Model                                              | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|----------------------------------------------------|--------------------------|-------------------------|------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                 | still failing            | 2026-05-04 20:21:24 BST | 3/3 recent runs failed |
| `facebook/pe-av-large`                             | still failing            | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`          | still failing            | 2026-02-13 13:18:47 GMT | 3/3 recent runs failed |
| `mlx-community/LFM2-VL-1.6B-8bit`                  | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16`                 | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 13
- **Summary diagnostics models:** 42
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1199.23s (1199.23s)
- **Average runtime per model:** 21.80s (21.80s)
- **Dominant runtime phase:** upstream prefill / first-token dominated 14/55 measured model runs (54% of tracked runtime).
- **Phase totals:** model load=111.79s, local prompt prep=0.17s, upstream prefill / first-token=639.24s, post-prefill decode=425.90s, generation total (unsplit)=4.13s, cleanup=8.37s
- **Generation total:** 1069.26s across 51 model(s); upstream prefill / first-token split available for 48/51 model(s).
- **Observed stop reasons:** completed=29, exception=7, max_tokens=19
- **Validation overhead:** 14.05s total (avg 0.26s across 55 model(s)).
- **Upstream prefill / first-token latency:** Avg 13.32s | Min 0.09s | Max 88.56s across 48 model(s).
- **What this likely means:** Most measured runtime is spent inside upstream generation before the first token is available.
- **Suggested next action:** Inspect prompt/image token accounting, dynamic-resolution image burden, prefill step sizing, and cache/prefill behavior.

---

## Models Not Flagged (42 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 5 model(s).
- **Ran, but with quality warnings:** 37 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260509-165009_DSC09954.jpg --exclude mlx-community/Qwen3-VL-2B-Thinking-bf16 Qwen/Qwen3-VL-2B-Instruct --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `facebook/pe-av-large`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/MolmoPoint-8B-fp16`, `microsoft/Phi-3.5-vision-instruct`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/llava-v1.6-mistral-7b-8bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/Qwen3.5-9B-MLX-4bit`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260509-165009_DSC09954.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-05-10 00:41:07 BST by [check_models](https://github.com/jrp2014/check_models)._

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260510+84961223 |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.8.0                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.14.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.4.0               |
| macOS Version   | 26.4.1                      |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |

