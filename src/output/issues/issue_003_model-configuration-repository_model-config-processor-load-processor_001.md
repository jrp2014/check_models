<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[model configuration/repository\]\[Model config: Processor load / processor error\] Processor config is missing image processor affecting 2 model(s)

## Summary

2 model(s) show **Model config: Processor load / processor error** that should be filed against model configuration / repository.

- **Observed problem:** Processor config is missing image processor
- **Target:** model configuration / repository
- **Affected models:** 2
- **Fixed when:** Load/generation completes or fails with a narrower owner.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                           | Observed Behavior                                                       | Token Counts   | Optional Context                                                                                                                                                                                             |
|-------------------------------------------------|-------------------------------------------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`  | Loaded processor has no image_processor; expected multimodal processor. | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260703T230155Z_007_mlx-community_diffusiongemma-26B-A4B-it-8bit_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_49.json)  |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` | Loaded processor has no image_processor; expected multimodal processor. | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260703T230155Z_008_mlx-community_diffusiongemma-26B-A4B-it-mxfp8_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_0c.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/diffusiongemma-26B-A4B-it-8bit` fails with: Loaded processor has no image_processor; expected multimodal processor.
- Root exception: `builtins.ValueError`: Loaded processor has no image_processor; expected multimodal processor.
- `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` fails with: Loaded processor has no image_processor; expected multimodal processor.
- Root exception: `builtins.ValueError`: Loaded processor has no image_processor; expected multimodal processor.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.
Use a local copy of `cats.jpg` or replace it with an equivalent test image.
Image SHA256: `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/diffusiongemma-26B-A4B-it-8bit --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/diffusiongemma-26B-A4B-it-mxfp8 --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'mlx-community/diffusiongemma-26B-A4B-it-8bit'
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
  "model": "mlx-community/diffusiongemma-26B-A4B-it-8bit"
}
```

Optional advanced context:

- `mlx-community/diffusiongemma-26B-A4B-it-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260703T230155Z_007_mlx-community_diffusiongemma-26B-A4B-it-8bit_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_49.json)
- `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260703T230155Z_008_mlx-community_diffusiongemma-26B-A4B-it-mxfp8_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_0c.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect `preprocessor_config.json`, `processor_config.json`, and AutoProcessor mapping.
- [ ] Verify the loaded processor exposes the image processor expected by mlx-vlm.
- [ ] Check whether the model repo needs processor files or mlx-vlm needs a fallback path.
- [ ] Reproduce with the single affected model before judging output quality.


## Appendix: Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.3                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260703+de7b4ed9                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
| transformers               | 5.12.1                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.22.0                                                                                                                                                   |
| Python Version             | 3.13.13                                                                                                                                                  |
| OS                         | Darwin 25.5.0                                                                                                                                            |
| macOS Version              | 26.5.2                                                                                                                                                   |
| SDK Version                | 26.5                                                                                                                                                     |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                                       |
| Xcode Version              | 26.6                                                                                                                                                     |
| Xcode Build                | 17F113                                                                                                                                                   |
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
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,451,352 bytes, sha256=7e5c9a3a3225bf3b04a5fe67c50602975d3698a45e2113433465848af47fd70c) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,747,136 bytes, sha256=6ae55e0952bcc7c19fd80f6f62dd1d0158448e0a58ce92384445c8f333f352ee)  |
| RAM                        | 128.0 GB                                                                                                                                                 |


## Appendix: Detailed Evidence

### `mlx-community/diffusiongemma-26B-A4B-it-8bit`

Observed error:

```text
Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

Root exception:

```text
builtins.ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

Traceback tail:

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

### `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

Observed error:

```text
Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-mxfp8: Loaded processor has no image_processor; expected multimodal processor.
```

Root exception:

```text
builtins.ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

Traceback tail:

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-mxfp8: Loaded processor has no image_processor; expected multimodal processor.
```

