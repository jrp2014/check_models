# Diagnostics Report — 2 failure(s), 7 harness issue(s) (mlx-vlm 0.3.12)

## Summary

Automated benchmarking of **46 locally-cached VLM models** found **2 hard failure(s)** and **7 harness/integration issue(s)** plus **2 preflight compatibility warning(s)** in successful models. 44 of 46 models succeeded.

Test image: `20260221-161455_DSC09294_DxO.jpg` (52.8 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [mlx-vlm]** Failed to process inputs with error: can only concatenate str (not "NoneType") to str (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 7 model(s). Next: validate stop-token decoding and long-context behavior.
- **[Medium] [mlx-vlm / mlx]** Stack-signal anomalies on 2 successful model(s). Next: inspect prompt/output token ratios and empty-output patterns.
- **[Medium] [mlx-vlm, transformers]** Preflight compatibility warnings (2 issue(s)). Next: verify dependency/version compatibility before model runs.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **Medium** | Failed to process inputs with error: can only concatenate str (not "N... | 1 (Florence-2-large-ft) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (deepseek-vl2-8bit) | `model configuration/repository` | verify model config, tokenizer files, and revision alignment. |
| **Medium** | Harness/integration | 7 (SmolVLM-Instruct, Qwen3-VL-2B-Instruct, Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, SmolVLM-Instruct-bf16, X-Reasoner-7B-8bit, Florence-2-large-ft) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Preflight compatibility warning | 2 issue(s) | `mlx-vlm, transformers` | verify dependency/version compatibility before model runs. |

---

## 1. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: can only concatenate str (not "NoneType") to str
**Owner (likely component):** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Affected model:** `microsoft/Florence-2-large-ft`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | Failed to process inputs with error: can only concatenate str (not "NoneType") to str | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft`
- Portable dependency probe: `python -m pip show mlx mlx-vlm mlx-lm transformers huggingface-hub tokenizers`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `microsoft/Florence-2-large-ft`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 876, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 862, in process_inputs
    return process_method(**args)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 163, in __call__
    image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 50, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 860, in preprocess
    self._validate_preprocess_kwargs(**kwargs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 823, in _validate_preprocess_kwargs
    validate_fast_preprocess_arguments(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        do_rescale=do_rescale,
        ^^^^^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        data_format=data_format,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 106, in validate_fast_preprocess_arguments
    raise ValueError("Only returning PyTorch tensors is currently supported.")
ValueError: Only returning PyTorch tensors is currently supported.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 889, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 862, in process_inputs
    return process_method(**args)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 185, in __call__
    self.image_token * self.num_image_tokens
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + self.tokenizer.bos_token
    ^~~~~~~~~~~~~~~~~~~~~~~~~~
TypeError: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 598, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 464, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1101, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 899, in process_inputs_with_fallback
    raise ValueError(
        f"Failed to process inputs with error: {fallback_error}"
    ) from fallback_error
ValueError: Failed to process inputs with error: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate str (not "NoneType") to str
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '1', '-', '1', '6', '1', '4', '5', '5', '_', 'D', 'S', 'C', '0', '9', '2', '9', '4', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: Analyze this image for cataloguing metadata.

Return exactly these three sections:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.

Context: Existing metadata hints (use only if visually consistent):
- Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK On a late winter afternoon, a heavy mist hangs over the hills surrounding Loch Katrine in Stronachlachar, Scotland. The muted colours of the February landscape are dominated by dark evergreen forests and the bare branches of deciduous trees. Nestled on the shore is a utility building, part of the infrastructure for the Loch Katrine aqueduct scheme. In th...
- Capture metadata: Taken on 2026-02-21 16:14:55 GMT (at 16:14:55 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 31254.13it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 2. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Loaded processor has no image_processor; expected multimodal processor.
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `mlx-community/deepseek-vl2-8bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/deepseek-vl2-8bit` | Loaded processor has no image_processor; expected multimodal processor. | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit`
- Portable dependency probe: `python -m pip show mlx mlx-vlm mlx-lm transformers huggingface-hub tokenizers`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/deepseek-vl2-8bit`

Traceback:

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

Captured stdout/stderr:

```text
=== STDOUT ===
Add pad token = ['<｜▁pad▁｜>'] to the tokenizer
<｜▁pad▁｜>:2
Add image token = ['<image>'] to the tokenizer
<image>:128815
Added grounding-related tokens
Added chat tokens

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]
Fetching 13 files: 100%|##########| 13/13 [00:00<00:00, 96335.60it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

---

## Preflight Compatibility Warnings (2 issue(s))

These warnings were detected before inference. They are non-fatal but should be tracked as potential upstream compatibility issues.

- `mlx-vlm load_image() has an unguarded URL startswith() branch; Path/BytesIO inputs can raise AttributeError in upstream code.`
  - Owner: `mlx-vlm`; suggested tracker: `mlx-vlm` (<https://github.com/ml-explore/mlx-vlm/issues/new>)
  - Suggested next action: check processor/chat-template wiring and generation kwargs.
- `transformers import utils no longer reference known backend guard env vars (TRANSFORMERS_NO_* / USE_*); check_models backend guard hints may be ignored with this version.`
  - Owner: `transformers`; suggested tracker: `transformers` (<https://github.com/huggingface/transformers/issues/new>)
  - Suggested next action: verify API compatibility and pinned version floor.

---

## Harness/Integration Issues (7 model(s))

These models completed successfully but show integration problems (for example stop-token leakage, decoding artifacts, or long-context breakdown) that likely point to stack/runtime behavior rather than inherent model quality limits.

### `HuggingFaceTB/SmolVLM-Instruct`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,451, output=12, output/prompt=0.83%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
Loch katrine, scotland, uk.
```

### `Qwen/Qwen3-VL-2B-Instruct`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,478, output=500, output/prompt=3.03%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16478 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
Title: Winter mist over Loch Katrine, Scotland

Description: A misty, overcast day at Loch Katrine in Stronachlachar, Scotland, with a small utility building on the shore. Tall electricity pylons and ...
```

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,052, output=500, output/prompt=47.53%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output switched language/script unexpectedly.

**Sample output:**

```text
Title: Misty Loch Katrine in Stronachlachar

Description: A serene scene of Loch Katrine in Stronachlachar, Scotland, with a heavy mist enveloping the hills. The landscape is characterized by a mix of...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,337, output=198, output/prompt=8.47%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 108 occurrences).

**Sample output:**

```text
Title:ĠMistyĠLochĠKatrineĠwithĠUtilityĠBuildingĊĊDescription:ĠAĠutilityĠbuildingĠstandsĠonĠtheĠshoreĠofĠLochĠKatrine,ĠsurroundedĠbyĠmistyĠhillsĠandĠevergreenĠforests.ĠPowerĠlinesĠstretchĠacrossĠtheĠsc...
```

### `mlx-community/SmolVLM-Instruct-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,451, output=12, output/prompt=0.83%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
Loch katrine, scotland, uk.
```

### `mlx-community/X-Reasoner-7B-8bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,489, output=500, output/prompt=3.03%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16489 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
**Title:** Remote Utility Building on Loch Katrine Shore

**Description:** A small white utility building sits on the shore of a calm, misty lake, surrounded by dense evergreen trees and a distant pow...
```

### `prince-canuma/Florence-2-large-ft`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=827, output=500, output/prompt=60.46%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;s&gt; appeared in generated text.
- Model output may not follow prompt or image contents.
- Output contains corrupted or malformed text segments.
- Output switched language/script unexpectedly.
- Output formatting deviated from the requested structure.

**Sample output:**

```text
<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s...
```

---

## Potential Stack Issues (2 model(s))

These models technically succeeded, but token/output patterns suggest likely integration/runtime issues worth checking upstream.

| Model | Prompt Tok | Output Tok | Output/Prompt | Symptom | Owner |
| ----- | ---------- | ---------- | ------------- | ------- | -------------- |
| `Qwen/Qwen3-VL-2B-Instruct` | 16,478 | 500 | 3.03% | Repetition under extreme prompt length | `mlx-vlm / mlx` |
| `mlx-community/X-Reasoner-7B-8bit` | 16,489 | 500 | 3.03% | Repetition under extreme prompt length | `mlx-vlm / mlx` |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** `mlx-community/InternVL3-14B-8bit`, `mlx-community/InternVL3-8B-bf16`

| Model | Status vs Previous Run | First Seen Failing | Recent Repro |
| ----- | ---------------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/deepseek-vl2-8bit` | still failing | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.3.12 |
| mlx | 0.30.7.dev20260223+d4c81062 |
| mlx-lm | 0.30.8 |
| transformers | 5.2.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.4.1 |
| Python Version | 3.13.11 |
| OS | Darwin 25.3.0 |
| macOS Version | 26.3 |
| GPU/Chip | Apple M4 Max |
| GPU Cores | 40 |
| Metal Support | Metal 4 |
| RAM | 128.0 GB |

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose
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

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit
```

### Prompt Used

```text
Analyze this image for cataloguing metadata.

Return exactly these three sections:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.

Context: Existing metadata hints (use only if visually consistent):
- Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK On a late winter afternoon, a heavy mist hangs over the hills surrounding Loch Katrine in Stronachlachar, Scotland. The muted colours of the February landscape are dominated by dark evergreen forests and the bare branches of deciduous trees. Nestled on the shore is a utility building, part of the infrastructure for the Loch Katrine aqueduct scheme. In th...
- Capture metadata: Taken on 2026-02-21 16:14:55 GMT (at 16:14:55 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-02-23 21:14:50 GMT by [check_models](https://github.com/jrp2014/check_models)._
