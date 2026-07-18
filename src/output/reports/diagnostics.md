<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 6 failure(s), 2 harness issue(s), 0 text-sanity issue(s) (mlx-vlm 0.6.5)

**Run summary:** 61 locally-cached VLM model(s) checked; 6 hard failure(s), 2 harness/integration issue(s), 0 text-sanity/semantic issue(s), 0 preflight warning(s), 55 successful run(s).

Test image: `cats.jpg` (0.2 MB).

## Impact and Run Conditions

- _Models tested:_ 61
- _Crashed:_ 6
- _Completed:_ 55
- _Prompt:_ Describe this image briefly.
- _Image:_ cats.jpg

<!-- markdownlint-disable MD060 -->

## Crash / Failure Matrix

| Model                                    | Outcome               | Phase      | Stage       | Primary exception                                                    | Suspected owner   |
|------------------------------------------|-----------------------|------------|-------------|----------------------------------------------------------------------|-------------------|
| mlx-community/FastVLM-0.5B-bf16          | Task outcome: crashed | model_load | Model Error | RemoteProtocolError: Server disconnected without sending a response. | mlx-vlm (medium)  |
| mlx-community/GLM-4.6V-Flash-6bit        | Task outcome: crashed | model_load | Model Error | RemoteProtocolError: Server disconnected without sending a response. | mlx-vlm (medium)  |
| mlx-community/Qwen3.5-9B-MLX-4bit        | Task outcome: crashed | model_load | Model Error | RemoteProtocolError: Server disconnected without sending a response. | mlx-vlm (medium)  |
| mlx-community/Qwen3.6-27B-mxfp8          | Task outcome: crashed | model_load | Model Error | RemoteProtocolError: Server disconnected without sending a response. | mlx-vlm (medium)  |
| mlx-community/SmolVLM-Instruct-bf16      | Task outcome: crashed | model_load | Model Error | RemoteProtocolError: Server disconnected without sending a response. | mlx-vlm (medium)  |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | Task outcome: crashed | model_load | Model Error | RemoteProtocolError: Server disconnected without sending a response. | mlx-vlm (medium)  |

<details>
<summary>Crash evidence: mlx-community/FastVLM-0.5B-bf16</summary>

```text
RemoteProtocolError: Server disconnected without sending a response.
```

```text
RemoteProtocolError: Server disconnected without sending a response.
ValueError: Model loading failed: Server disconnected without sending a response.
```

```text
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.
```

</details>

<details>
<summary>Crash evidence: mlx-community/GLM-4.6V-Flash-6bit</summary>

```text
RemoteProtocolError: Server disconnected without sending a response.
```

```text
RemoteProtocolError: Server disconnected without sending a response.
ValueError: Model loading failed: Server disconnected without sending a response.
```

```text
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.
```

</details>

<details>
<summary>Crash evidence: mlx-community/Qwen3.5-9B-MLX-4bit</summary>

```text
RemoteProtocolError: Server disconnected without sending a response.
```

```text
RemoteProtocolError: Server disconnected without sending a response.
ValueError: Model loading failed: Server disconnected without sending a response.
```

```text
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.
```

</details>

<details>
<summary>Crash evidence: mlx-community/Qwen3.6-27B-mxfp8</summary>

```text
RemoteProtocolError: Server disconnected without sending a response.
```

```text
RemoteProtocolError: Server disconnected without sending a response.
ValueError: Model loading failed: Server disconnected without sending a response.
```

```text
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.
```

</details>

<details>
<summary>Crash evidence: mlx-community/SmolVLM-Instruct-bf16</summary>

```text
RemoteProtocolError: Server disconnected without sending a response.
```

```text
RemoteProtocolError: Server disconnected without sending a response.
ValueError: Model loading failed: Server disconnected without sending a response.
```

```text
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.
```

</details>

<details>
<summary>Crash evidence: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

```text
RemoteProtocolError: Server disconnected without sending a response.
```

```text
RemoteProtocolError: Server disconnected without sending a response.
ValueError: Model loading failed: Server disconnected without sending a response.
```

```text
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.
```

</details>
<!-- markdownlint-enable MD060 -->
## mlx-vlm / MLX Issue Matrix

Routing labels reflect evidence confidence only; a crash remains a conclusive
failed task at every confidence level.
<!-- markdownlint-disable MD060 -->

| Target                                         | Problem                                                                           | Evidence Snapshot                                                                                                                                                                       | Affected Models                              | Confidence   | Evidence Type   | Fixed When                                                |
|------------------------------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|--------------|-----------------|-----------------------------------------------------------|
| `mlx-vlm`                                      | mlx-vlm: Model load / model error: Server disconnected without sending a response | Model Error \| phase model_load \| RemoteProtocolError \| 6 model cluster                                                                                                               | 6: `mlx-community/FastVLM-0.5B-bf16` (+5)    | confirmed    | failure         | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | Stop/control tokens leaked into generated text                                    | decoded text contains control token &lt;/think&gt; \| prompt=317 \| output/prompt=59.31% \| stop=completed                                                                              | 1: `mlx-community/Qwen3-VL-2B-Thinking-bf16` | confirmed    | harness         | No leaked stop/control tokens.                            |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short                             | generated_tokens~3 \| prompt_tokens=4103, output_tokens=3, output/prompt=0.1%, weak text=truncated \| prompt=4,103 \| output/prompt=0.07% \| visual input burden=100% \| stop=completed | 1: `mlx-community/paligemma2-3b-pt-896-4bit` | confirmed    | context_budget  | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->
---

## Issue 1: mlx-vlm — mlx-vlm: Model load / model error


### Expected and Actual Behaviour

- _Affected models:_ mlx-community/FastVLM-0.5B-bf16,
  mlx-community/GLM-4.6V-Flash-6bit, mlx-community/Qwen3.5-9B-MLX-4bit,
  mlx-community/Qwen3.6-27B-mxfp8, mlx-community/SmolVLM-Instruct-bf16,
  mlx-community/SmolVLM2-2.2B-Instruct-mlx
- _Expected:_ Affected reruns complete model load and generation, or fail with
  a narrower configuration/compatibility error that points to the owning
  layer.
- _Actual:_ mlx-vlm: Model load / model error: Server disconnected without
  sending a response
- _Filing target:_ mlx-vlm

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/FastVLM-0.5B-bf16 --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`


### Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

---

## Issue 2: mlx-vlm — Stop-token leakage

<!-- markdownlint-disable MD033 -->

### Generated Output Evidence

<details>
<summary>Complete generated output: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

```text
So, let's see. The image shows two cats lying on a pink couch. The couch is a bright pink color, and there are two remote controls next to the cats. One cat is on the left, with a striped pattern, and the other is on the right, also striped but maybe a bit different. Both cats are relaxed, lying down, maybe sleeping. The remotes are white with some buttons, and one has a blue button. The background is the pink couch, so the main elements are the two cats, the remotes, and the pink fabric. I need to describe this briefly.
</think>

Two tabby cats are lying on a bright pink couch, relaxed and asleep. A white remote control with a blue button sits between them, and another remote control is positioned near the left cat. The scene is cozy, with the cats' striped fur and the vibrant pink fabric of the couch forming the backdrop.
```

</details>
<!-- markdownlint-enable MD033 -->

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/Qwen3-VL-2B-Thinking-bf16
- _Expected:_ Affected reruns contain no leaked stop/control tokens and
  terminate cleanly before the configured max-token cap when the response is
  complete.
- _Actual:_ Stop/control tokens leaked into generated text
- _Filing target:_ mlx-vlm

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3-VL-2B-Thinking-bf16 --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`


### Expected Fix Signal

- [ ] Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

---

## Issue 3: mlx-vlm first; MLX if cache/runtime reproduces — Long-context collapse

<!-- markdownlint-disable MD033 -->

### Generated Output Evidence

<details>
<summary>Complete generated output: mlx-community/paligemma2-3b-pt-896-4bit</summary>

```text
Cat.
```

</details>
<!-- markdownlint-enable MD033 -->

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/paligemma2-3b-pt-896-4bit
- _Expected:_ A same-command rerun and a reduced image/text burden rerun show
  consistent prompt-token accounting and no long-context collapse.
- _Actual:_ Long-context generation collapsed or became too short
- _Filing target:_ mlx-vlm first; MLX if cache/runtime reproduces

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/paligemma2-3b-pt-896-4bit --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`


### Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

<!-- markdownlint-disable MD060 -->

## Prompt / Input Burden Evidence

| Model                                   | Burden       |   Total tokens |   Text estimate |   Non-text estimate | Source            |
|-----------------------------------------|--------------|----------------|-----------------|---------------------|-------------------|
| mlx-community/paligemma2-3b-pt-896-4bit | visual_input |           4103 |               6 |                4097 | estimated_nontext |
<!-- markdownlint-enable MD060 -->
<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                         |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.5                                                                                                                                           |
| mlx                        | 0.32.1.dev20260718+b7c3dd6d                                                                                                                     |
| mlx-lm                     | 0.31.3                                                                                                                                          |
| mlx-audio                  | 0.4.5                                                                                                                                           |
| transformers               | 5.14.1                                                                                                                                          |
| tokenizers                 | 0.22.2                                                                                                                                          |
| huggingface-hub            | 1.24.0                                                                                                                                          |
| Python Version             | 3.13.13                                                                                                                                         |
| OS                         | Darwin 25.5.0                                                                                                                                   |
| macOS Version              | 26.5.2                                                                                                                                          |
| SDK Version                | 26.5                                                                                                                                            |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                              |
| Xcode Version              | 26.6                                                                                                                                            |
| Xcode Build                | 17F113                                                                                                                                          |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                      |
| Metal SDK                  | MacOSX26.5.sdk                                                                                                                                  |
| Metal Compiler Version     | Apple metal version 32023.883 (metalfe-32023.883)                                                                                               |
| Metallib Linker Version    | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                  |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                 |
| GPU/Chip                   | Apple M5 Max                                                                                                                                    |
| GPU Cores                  | 40                                                                                                                                              |
| Metal Support              | Metal 4                                                                                                                                         |
| MLX Install Type           | editable local source                                                                                                                           |
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,449,848 bytes, sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=c03f8b1be065c164ed6c9380cc87b205f1c11c0e5b0cda028984159c57c79a3b)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD060 -->
