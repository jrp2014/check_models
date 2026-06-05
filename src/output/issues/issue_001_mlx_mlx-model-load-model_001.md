<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx\]\[MLX: Model load / model error\] Weight/config mismatch during model load affecting 1 model(s)

## Summary

1 model(s) show **MLX: Model load / model error** that should be filed against mlx.

- **Observed problem:** Weight/config mismatch during model load
- **Target:** mlx
- **Affected models:** 1
- **Fixed when:** Load/generation completes or fails with a narrower owner.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                     | Observed Behavior                                                                                                                                                                               | Token Counts   | Optional Context                                                                                                                                                                                |
|-------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | Received 4 parameters not in model: multi_modal_projector.linear_1.biases, multi_modal_projector.linear_1.scales, multi_modal_projector.linear_2.biases, multi_modal_projector.linear_2.scales. | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_007_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/Kimi-VL-A3B-Thinking-8bit` fails with: Model loading failed: Received 4 parameters not in model: multi_modal_projector.linear_1.biases, multi_modal_projector.linear_1.scales, multi_modal_projector.linear_2.biases, multi_modal_projector.linear_2.scales.
- Root exception: `builtins.ValueError`: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_projector.linear_1.scales,<br>multi_modal_projector.linear_2.biases,<br>multi_modal_projector.l...


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Kimi-VL-A3B-Thinking-8bit --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/Kimi-VL-A3B-Thinking-8bit'
IMAGE = '/Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg'
PROMPT = 'Describe this image briefly.'
LOAD_KWARGS = {'trust_remote_code': True}
GENERATE_KWARGS = {'max_tokens': 200, 'temperature': 0.0, 'prefill_step_size': 4096}
model, processor = load(MODEL, **LOAD_KWARGS)
result = generate(model, processor, PROMPT, image=IMAGE, **GENERATE_KWARGS)
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
  "image": "/Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/Kimi-VL-A3B-Thinking-8bit"
}
```

Optional advanced context:

- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_007_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Compare checkpoint keys with the selected model class and model config.
- [ ] Inspect missing/unexpected projector, scale, bias, and quantized-weight parameter names.
- [ ] Verify the model repo revision matches the mlx-vlm/mlx loader expectations.
- [ ] Reproduce after upgrading/downgrading mlx-vlm and mlx to isolate version compatibility.


## Appendix: Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.1                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260605+6ea7a00d                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.3                                                                                                                                                    |
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
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,675,136 bytes, sha256=7c19dc6a8bf6b56db0155bed7a1d2b0a54182b42eba2fcf0f42b97d3d4c57eff)  |
| RAM                        | 128.0 GB                                                                                                                                                 |


## Appendix: Detailed Evidence

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

Observed error:

```text
Model loading failed: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

Root exception:

```text
builtins.ValueError: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

Traceback tail:

```text
Traceback (most recent call last):
ValueError: Model loading failed: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

