# Diagnostics Report — 7 failure(s), 2 harness issue(s) (mlx-vlm 0.5.0)

**Run summary:** 55 locally-cached VLM model(s) checked; 7 hard failure(s), 2 harness/integration issue(s), 0 preflight warning(s), 48 successful run(s).

Test image: `20260502-173345_DSC09912_DxO.jpg` (27.3 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

| Target                                         | Problem                                                                                              | Evidence Snapshot                                                                                                                                               | Affected Models                                            | Issue Draft                                                                                                                                                              | Evidence Bundle                                                                                                                                                                                          | Fixed When                                                |
|------------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                                          | Weight/config mismatch during model load                                                             | Model Error \| phase model_load \| ValueError \| 2 model cluster                                                                                                | 2: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (+1)                 | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)                                             | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T112302Z_001_LiquidAI_LFM2.5-VL-450M-MLX-bf16_MLX_MODEL_LOAD_MODEL_853049863f38.json)                 | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | mlx-vlm: Decode / runtime error: property 'text' of 'NaiveStreamingDetokenizer' object has no setter | Error \| phase decode \| AttributeError \| 3 model cluster                                                                                                      | 3: `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (+2) | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_mlx-vlm-decode-error_001.md)                                         | [3 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T112302Z_004_mlx-community_ERNIE-4.5-VL-28B-A3B-Thinking-bf16_MLX_VLM_DECODE_ERROR_c6f291b6246e.json) | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | Missing module/import during model load                                                              | Model Error \| phase model_load \| ValueError                                                                                                                   | 1: `facebook/pe-av-large`                                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm_mlx-vlm-model-load-model_001.md)                                     | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T112302Z_002_facebook_pe-av-large_MLX_VLM_MODEL_LOAD_MODEL_8b244da8c605.json)                            | Load/generation completes or fails with a narrower owner. |
| model configuration / repository               | Processor config is missing image processor                                                          | Processor Error \| phase processor_load \| ValueError                                                                                                           | 1: `mlx-community/MolmoPoint-8B-fp16`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_model-configuration-repository_model-config-processor-load-processor_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T112302Z_008_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json)             | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | Tokenizer decode leaked BPE/byte markers                                                             | 61 BPE space markers found in decoded text \| prompt=3,619 \| output/prompt=2.98% \| nontext burden=88% \| stop=completed                                       | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm_encoding_001.md)                                                     | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T112302Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json)            | No BPE/byte markers in output.                            |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short                                                | output/prompt=0.1% \| prompt_tokens=16901, output_tokens=11, output/prompt=0.1% \| prompt=16,901 \| output/prompt=0.07% \| nontext burden=97% \| stop=completed | 1: `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm-mlx_long-context_001.md)                                             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T112302Z_009_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)                  | Full and reduced reruns avoid context collapse.           |

---

## Upstream Filing Notes

File one upstream issue per row above. The linked issue drafts are the
pasteable bodies; this diagnostics file is the compact run-level queue.

- **Issue drafts:** 6 root-cause cluster(s).
- **Suggested targets:** `mlx`=1, `mlx-vlm`=3, `mlx-vlm / mlx`=1, `model configuration/repository`=1.
- **Standalone evidence:** each issue draft includes minimal inline evidence plus a native upstream repro command before any appendix detail.
- **Supporting files:** `repro JSON` links point to the canonical GitHub copies of the bundles with prompt, environment, and generated-output context. If you are filing from an uncommitted ad-hoc run, attach the JSON manually.

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
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | Model loading failed: Received 4 parameters not in model: multi_modal_projector.linear_1.biases, multi_modal_projector.linear_1.scales, multi_modal_projector.linear_2.biases, multi_modal_projector.linear_2.scales. | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |

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

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ runtime_failure \| MLX_VLM_DECODE_ERROR
- _Summary:_ property 'text' of 'NaiveStreamingDetokenizer' object has no
  setter
- _Evidence:_ stage=Error \| phase=decode \| code=MLX_VLM_DECODE_ERROR \|
  type=builtins.AttributeError
- _Token context:_ stop=exception
- _Next action:_ Inspect prompt-template, stop-token, and decode
  post-processing behavior.

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
- _Classification:_ runtime_failure \| MLX_VLM_DECODE_ERROR
- _Summary:_ property 'text' of 'NaiveStreamingDetokenizer' object has no
  setter
- _Evidence:_ stage=Error \| phase=decode \| code=MLX_VLM_DECODE_ERROR \|
  type=builtins.AttributeError
- _Token context:_ stop=exception
- _Next action:_ Inspect prompt-template, stop-token, and decode
  post-processing behavior.

| Model                             | Observed Behavior                                                                                                                              | First Seen Failing      | Recent Repro           |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/LFM2-VL-1.6B-8bit` | Model runtime error during generation for mlx-community/LFM2-VL-1.6B-8bit: property 'text' of 'NaiveStreamingDetokenizer' object has no setter | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |

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
- _Classification:_ runtime_failure \| MLX_VLM_DECODE_ERROR
- _Summary:_ property 'text' of 'NaiveStreamingDetokenizer' object has no
  setter
- _Evidence:_ stage=Error \| phase=decode \| code=MLX_VLM_DECODE_ERROR \|
  type=builtins.AttributeError
- _Token context:_ stop=exception
- _Next action:_ Inspect prompt-template, stop-token, and decode
  post-processing behavior.

| Model                               | Observed Behavior                                                                                                                                | First Seen Failing      | Recent Repro           |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16` | Model runtime error during generation for mlx-community/LFM2.5-VL-1.6B-bf16: property 'text' of 'NaiveStreamingDetokenizer' object has no setter | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |

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
- _Evidence:_ 61 BPE space markers found in decoded text
- _Token context:_ prompt=3,619 \| output/prompt=2.98% \| nontext burden=88%
  \| stop=completed
- _Next action:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 61 occurrences).
- Model output may not follow prompt or image contents (missing: vast, expanse, adorned, small, floats).
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
- _Evidence:_ output/prompt=0.1% \| prompt_tokens=16901, output_tokens=11,
  output/prompt=0.1%
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

| Model                                              | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|----------------------------------------------------|--------------------------|-------------------------|------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                 | still failing            | 2026-05-04 20:21:24 BST | 3/3 recent runs failed |
| `facebook/pe-av-large`                             | still failing            | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`          | still failing            | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |
| `mlx-community/LFM2-VL-1.6B-8bit`                  | still failing            | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                | still failing            | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16`                 | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 9
- **Summary diagnostics models:** 46
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1249.93s (1249.93s)
- **Average runtime per model:** 22.73s (22.73s)
- **Dominant runtime phase:** upstream prefill / first-token dominated 16/55 measured model runs (52% of tracked runtime).
- **Phase totals:** model load=107.42s, local prompt prep=0.16s, upstream prefill / first-token=641.95s, post-prefill decode=481.53s, generation total (unsplit)=3.70s, cleanup=6.02s
- **Generation total:** 1127.18s across 51 model(s); upstream prefill / first-token split available for 48/51 model(s).
- **Observed stop reasons:** completed=29, exception=7, max_tokens=19
- **Validation overhead:** 11.28s total (avg 0.21s across 55 model(s)).
- **Upstream prefill / first-token latency:** Avg 13.37s | Min 0.08s | Max 104.99s across 48 model(s).
- **What this likely means:** Most measured runtime is spent inside upstream generation before the first token is available.
- **Suggested next action:** Inspect prompt/image token accounting, dynamic-resolution image burden, prefill step sizing, and cache/prefill behavior.

---

## Models Not Flagged (46 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 11 model(s).
- **Ran, but with quality warnings:** 35 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg --exclude mlx-community/Qwen3-VL-2B-Thinking-bf16 Qwen/Qwen3-VL-2B-Instruct --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `facebook/pe-av-large`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-05-09 19:01:23 BST by [check_models](https://github.com/jrp2014/check_models)._

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260509+84961223 |
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
