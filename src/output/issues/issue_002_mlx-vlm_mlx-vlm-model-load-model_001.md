<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm\]\[mlx-vlm: Model load / model error\] mlx-vlm: Model load / model error: property 'eos_token_id' of 'ModelConfig' object has no setter affecting 1 model(s)

## Summary

1 model(s) show **mlx-vlm: Model load / model error** that should be filed against mlx-vlm.

- **Observed problem:** mlx-vlm: Model load / model error: property 'eos_token_id' of 'ModelConfig' object has no setter
- **Target:** mlx-vlm
- **Affected models:** 1
- **Fixed when:** Load/generation completes or fails with a narrower owner.


## Model

- **Primary model:** `mlx-community/MolmoPoint-8B-fp16`
- **Affected model count:** 1
- **Revision:** `unknown`
- **Trust remote code:** `true`


## Inputs

- **Prompt:** `Describe this image briefly.`
- **Image:** `/Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg`
- **Image facts:** 640x480 (0.31 MP), JPG extension, 173131 bytes, sha256 `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`
- **Shareable:** unknown


## Expected Behavior

- The native `mlx-vlm` CLI/Python repro should load the model, process the prompt and image, and return a response without the observed failure or quality regression.
- If the model family is intentionally unsupported, `mlx-vlm` should fail before generation with a clear model-specific message.


## Actual Behavior

- mlx-vlm: Model load / model error: property 'eos_token_id' of 'ModelConfig' object has no setter
- Representative failure:

```text
Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter
```


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                              | Observed Behavior                                             | Token Counts   | Optional Context                                                                                                                                                                             |
|------------------------------------|---------------------------------------------------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/MolmoPoint-8B-fp16` | property 'eos_token_id' of 'ModelConfig' object has no setter | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260613T235453Z_006_mlx-community_MolmoPoint-8B-fp16_MLX_VLM_MODEL_LOAD_MODEL_7cbd53695717.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/MolmoPoint-8B-fp16` fails with: Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter
- Root exception: `builtins.AttributeError`: property 'eos_token_id' of 'ModelConfig' object has no setter


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/MolmoPoint-8B-fp16 --image /Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'mlx-community/MolmoPoint-8B-fp16'
IMAGE = '/Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg'
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
  "image": "/Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/MolmoPoint-8B-fp16"
}
```

Optional advanced context:

- `mlx-community/MolmoPoint-8B-fp16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260613T235453Z_006_mlx-community_MolmoPoint-8B-fp16_MLX_VLM_MODEL_LOAD_MODEL_7cbd53695717.json)
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

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.3                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260614+89064477                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
| transformers               | 5.12.0                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.19.0                                                                                                                                                   |
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
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (157,751,704 bytes, sha256=ba9913d81d92bbbde42bbc6dda27e80ecb31db6031fa073e6c8aeb0666d47c33) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,671,712 bytes, sha256=3a48ca2ae7659130de59374b0c50f6ba11b1dfecb8f6a2549f1b72fb41ac921c)  |
| RAM                        | 128.0 GB                                                                                                                                                 |


## Appendix: Detailed Evidence

### `mlx-community/MolmoPoint-8B-fp16`

Observed error:

```text
Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter
```

Root exception:

```text
builtins.AttributeError: property 'eos_token_id' of 'ModelConfig' object has no setter
```

Traceback tail:

```text
    setattr(model_config, key, config[key])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: property 'eos_token_id' of 'ModelConfig' object has no setter
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter
```

