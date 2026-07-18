<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm\]\[mlx-vlm: Model load / model error\] mlx-vlm: Model load / model error: Server disconnected without sending a response affecting 6 model(s)

## Summary

6 model(s) show **mlx-vlm: Model load / model error** that should be filed against mlx-vlm.

- **Observed problem:** mlx-vlm: Model load / model error: Server disconnected without sending a response
- **Target:** mlx-vlm
- **Affected models:** 6
- **Fixed when:** Load/generation completes or fails with a narrower owner.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                      | Observed Behavior                                                    | Token Counts   | Optional Context                                                                                                                                                                                     |
|--------------------------------------------|----------------------------------------------------------------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/FastVLM-0.5B-bf16`          | RemoteProtocolError: Server disconnected without sending a response. | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_001_mlx-community_FastVLM-0.5B-bf16_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json)          |
| `mlx-community/GLM-4.6V-Flash-6bit`        | RemoteProtocolError: Server disconnected without sending a response. | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_002_mlx-community_GLM-4.6V-Flash-6bit_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json)        |
| `mlx-community/Qwen3.5-9B-MLX-4bit`        | RemoteProtocolError: Server disconnected without sending a response. | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_003_mlx-community_Qwen3.5-9B-MLX-4bit_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json)        |
| `mlx-community/Qwen3.6-27B-mxfp8`          | RemoteProtocolError: Server disconnected without sending a response. | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_004_mlx-community_Qwen3.6-27B-mxfp8_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json)          |
| `mlx-community/SmolVLM-Instruct-bf16`      | RemoteProtocolError: Server disconnected without sending a response. | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_005_mlx-community_SmolVLM-Instruct-bf16_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json)      |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | RemoteProtocolError: Server disconnected without sending a response. | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_006_mlx-community_SmolVLM2-2.2B-Instruct-mlx_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/FastVLM-0.5B-bf16` fails with: RemoteProtocolError: Server disconnected without sending a response.
- Later exceptions: RemoteProtocolError: Server disconnected without sending a response. -> ValueError: Model loading failed: Server disconnected without sending a response.
- `mlx-community/GLM-4.6V-Flash-6bit` fails with: RemoteProtocolError: Server disconnected without sending a response.
- Later exceptions: RemoteProtocolError: Server disconnected without sending a response. -> ValueError: Model loading failed: Server disconnected without sending a response.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.
Use a local copy of `cats.jpg` or replace it with an equivalent test image.
Image SHA256: `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`

## Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/FastVLM-0.5B-bf16 --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'mlx-community/FastVLM-0.5B-bf16'
IMAGE = 'cats.jpg'
PROMPT = 'Describe this image briefly.'
LOAD_KWARGS = {'trust_remote_code': True}
GENERATE_KWARGS = {'max_tokens': 200, 'temperature': 0.0, 'prefill_step_size': 4096}
model, processor = load(MODEL, **LOAD_KWARGS)
formatted_prompt = apply_chat_template(
    processor,
    model.config,
    PROMPT,
    num_images=1,
)
if isinstance(formatted_prompt, list):
    formatted_prompt = "\n".join(str(message) for message in formatted_prompt)
result = generate(model, processor, formatted_prompt, image=IMAGE, **GENERATE_KWARGS)
print(result.text)
```

Prompt:

```text
Describe this image briefly.
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 200,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "cats.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/FastVLM-0.5B-bf16"
}
```

Optional advanced context:

- `mlx-community/FastVLM-0.5B-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_001_mlx-community_FastVLM-0.5B-bf16_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json)
- `mlx-community/GLM-4.6V-Flash-6bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_002_mlx-community_GLM-4.6V-Flash-6bit_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json)
- `mlx-community/Qwen3.5-9B-MLX-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_003_mlx-community_Qwen3.5-9B-MLX-4bit_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json)
- `mlx-community/Qwen3.6-27B-mxfp8`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_004_mlx-community_Qwen3.6-27B-mxfp8_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json)
- `mlx-community/SmolVLM-Instruct-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_005_mlx-community_SmolVLM-Instruct-bf16_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json)
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_006_mlx-community_SmolVLM2-2.2B-Instruct-mlx_MLX_VLM_MODEL_LOAD_MODEL_baf9616f8913.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect the exported error package, load phase, and traceback owner.
- [ ] Check model config, tokenizer files, and weight shape compatibility.
- [ ] Compare against installed mlx, mlx-vlm, mlx-lm, transformers, and tokenizers versions.
- [ ] Reproduce with the single affected model before judging output quality.


## Appendix: Environment

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


## Appendix: Detailed Evidence

### `mlx-community/FastVLM-0.5B-bf16`

Observed error:

```text
RemoteProtocolError: Server disconnected without sending a response.
```

Exception chain:

```text
RemoteProtocolError: Server disconnected without sending a response.
ValueError: Model loading failed: Server disconnected without sending a response.
```

Traceback tail:

```text
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.
```

### `mlx-community/GLM-4.6V-Flash-6bit`

Observed error:

```text
RemoteProtocolError: Server disconnected without sending a response.
```

Exception chain:

```text
RemoteProtocolError: Server disconnected without sending a response.
ValueError: Model loading failed: Server disconnected without sending a response.
```

Traceback tail:

```text
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.
```

### `mlx-community/Qwen3.5-9B-MLX-4bit`

Observed error:

```text
RemoteProtocolError: Server disconnected without sending a response.
```

Exception chain:

```text
RemoteProtocolError: Server disconnected without sending a response.
ValueError: Model loading failed: Server disconnected without sending a response.
```

Traceback tail:

```text
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 231, in _receive_event
    raise RemoteProtocolError(msg)
httpcore.RemoteProtocolError: Server disconnected without sending a response.
```

_Additional affected models are listed in the Affected Models table above._

