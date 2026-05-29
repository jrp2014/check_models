<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx\]\[MLX: Model load / weight/config mismatch\] Weight/config mismatch during model load affecting 1 model(s)

## Summary

1 model(s) show **MLX: Model load / weight/config mismatch** that should be filed against mlx.

- **Observed problem:** Weight/config mismatch during model load
- **Target:** mlx
- **Affected models:** 1
- **Fixed when:** Load/generation completes or fails with a narrower owner.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                               | Observed Behavior                                                                                     | Token Counts   | Optional Context                                                                                                                                                                                 |
|-------------------------------------|-------------------------------------------------------------------------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16` | Missing 2 parameters: multi_modal_projector.layer_norm.bias, multi_modal_projector.layer_norm.weight. | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260529T205941Z_008_mlx-community_LFM2.5-VL-1.6B-bf16_MLX_MODEL_LOAD_WEIGHT_MISMATCH_d9bf5051d.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/LFM2.5-VL-1.6B-bf16` fails with: Model loading failed: Missing 2 parameters: multi_modal_projector.layer_norm.bias, multi_modal_projector.layer_norm.weight.
- Root exception: `builtins.ValueError`: Missing 2 parameters: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_projector.layer_norm.weight.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/LFM2.5-VL-1.6B-bf16 --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/LFM2.5-VL-1.6B-bf16'
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
  "model": "mlx-community/LFM2.5-VL-1.6B-bf16"
}
```

Optional advanced context:

- `mlx-community/LFM2.5-VL-1.6B-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260529T205941Z_008_mlx-community_LFM2.5-VL-1.6B-bf16_MLX_MODEL_LOAD_WEIGHT_MISMATCH_d9bf5051d.json)
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

| Component                   | Version                                                                                                                                                                           |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                     | 0.5.0                                                                                                                                                                             |
| mlx                         | 0.31.2                                                                                                                                                                            |
| mlx-metal                   | 0.31.2                                                                                                                                                                            |
| mlx-lm                      | 0.31.3                                                                                                                                                                            |
| mlx-audio                   | 0.4.3                                                                                                                                                                             |
| transformers                | 5.9.0                                                                                                                                                                             |
| tokenizers                  | 0.22.2                                                                                                                                                                            |
| huggingface-hub             | 1.17.0                                                                                                                                                                            |
| Python Version              | 3.13.13                                                                                                                                                                           |
| OS                          | Darwin 25.5.0                                                                                                                                                                     |
| macOS Version               | 26.5                                                                                                                                                                              |
| SDK Version                 | 26.5                                                                                                                                                                              |
| SDK Path                    | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                                                                |
| Xcode Version               | 26.5                                                                                                                                                                              |
| Xcode Build                 | 17F42                                                                                                                                                                             |
| Active Developer Directory  | /Applications/Xcode.app/Contents/Developer                                                                                                                                        |
| Metal SDK                   | MacOSX26.5.sdk                                                                                                                                                                    |
| Metal Compiler Version      | Apple metal version 32023.883 (metalfe-32023.883)                                                                                                                                 |
| Metallib Linker Version     | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                                                    |
| Apple Clang Version         | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                                                   |
| GPU/Chip                    | Apple M5 Max                                                                                                                                                                      |
| GPU Cores                   | 40                                                                                                                                                                                |
| Metal Support               | Metal 4                                                                                                                                                                           |
| MLX Install Type            | wheel/site-packages                                                                                                                                                               |
| MLX Distribution Root       | /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                                                   |
| mlx-metal Distribution Root | /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                                                   |
| MLX Core Extension          | /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib                | /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/mlx/lib/mlx.metallib (157,748,008 bytes, sha256=8c8bfcece8c0610745b68879771e5aa1b92b29fa5e17172e5508e4f5153d8d15) |
| MLX libmlx.dylib            | /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/mlx/lib/libmlx.dylib (21,653,808 bytes, sha256=2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd)  |
| RAM                         | 128.0 GB                                                                                                                                                                          |


## Appendix: Detailed Evidence

### `mlx-community/LFM2.5-VL-1.6B-bf16`

Observed error:

```text
Model loading failed: Missing 2 parameters: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

Root exception:

```text
builtins.ValueError: Missing 2 parameters: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

Traceback tail:

```text
multi_modal_projector.layer_norm.weight.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: Missing 2 parameters: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

