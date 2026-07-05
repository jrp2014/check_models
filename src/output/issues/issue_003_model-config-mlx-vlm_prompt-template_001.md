<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[model-config / mlx-vlm\]\[Prompt-template / image-placeholder mismatch\] Prompt/template output shape mismatch affecting 5 model(s)

## Summary

5 model(s) show **Prompt-template / image-placeholder mismatch** that should be filed against model repo first; mlx-vlm if template handling disagrees.

- **Observed problem:** Prompt/template output shape mismatch
- **Target:** model repo first; mlx-vlm if template handling disagrees
- **Affected models:** 5
- **Fixed when:** Requested sections render without template leakage.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                     | Observed Behavior   | Token Counts                                                              | Optional Context                                                                                                                                                                                       |
|-------------------------------------------|---------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Qwen/Qwen3-VL-2B-Instruct`               | generated_tokens~3  | prompt=315 \| output/prompt=0.95% \| nontext burden=98% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_001_Qwen_Qwen3-VL-2B-Instruct_model_config_mlx_vlm_prompt_template_001.json)               |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16` | generated_tokens~3  | prompt=315 \| output/prompt=0.95% \| nontext burden=98% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_004_mlx-community_Qwen3-VL-2B-Instruct-bf16_model_config_mlx_vlm_prompt_template_001.json) |
| `mlx-community/Qwen3.5-35B-A3B-bf16`      | generated_tokens~3  | prompt=319 \| output/prompt=0.94% \| nontext burden=98% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_006_mlx-community_Qwen3.5-35B-A3B-bf16_model_config_mlx_vlm_prompt_template_001.json)      |
| `mlx-community/Qwen3.6-27B-mxfp8`         | generated_tokens~2  | prompt=319 \| output/prompt=0.63% \| nontext burden=98% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_007_mlx-community_Qwen3.6-27B-mxfp8_model_config_mlx_vlm_prompt_template_001.json)         |
| `mlx-community/X-Reasoner-7B-8bit`        | generated_tokens~3  | prompt=417 \| output/prompt=0.72% \| nontext burden=99% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_008_mlx-community_X-Reasoner-7B-8bit_model_config_mlx_vlm_prompt_template_001.json)        |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `Qwen/Qwen3-VL-2B-Instruct`: Output appears truncated to about 3 tokens.
- Output excerpt: `This image`
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: Output appears truncated to about 3 tokens.
- Output excerpt: `This image`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.
Use a local copy of `cats.jpg` or replace it with an equivalent test image.
Image SHA256: `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`

Native CLI:

```bash
python -m mlx_vlm.generate --model Qwen/Qwen3-VL-2B-Instruct --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3-VL-2B-Instruct-bf16 --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-35B-A3B-bf16 --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.6-27B-mxfp8 --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/X-Reasoner-7B-8bit --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'Qwen/Qwen3-VL-2B-Instruct'
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
  "model": "Qwen/Qwen3-VL-2B-Instruct"
}
```

Optional advanced context:

- `Qwen/Qwen3-VL-2B-Instruct`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_001_Qwen_Qwen3-VL-2B-Instruct_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_004_mlx-community_Qwen3-VL-2B-Instruct-bf16_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/Qwen3.5-35B-A3B-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_006_mlx-community_Qwen3.5-35B-A3B-bf16_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/Qwen3.6-27B-mxfp8`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_007_mlx-community_Qwen3.6-27B-mxfp8_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/X-Reasoner-7B-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_008_mlx-community_X-Reasoner-7B-8bit_model_config_mlx_vlm_prompt_template_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect chat template selection and rendered message roles.
- [ ] Verify image placeholder count and order match the processor config.
- [ ] Check EOS defaults and whether the template expects explicit assistant prefixes.


## Appendix: Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.4                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260704+de7b4ed9                                                                                                                              |
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
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,747,136 bytes, sha256=9d942d98a9a9f3e42b3f22c6606bc1ee621d28a9fb512d0cdba6edbb9ef79df8)  |
| RAM                        | 128.0 GB                                                                                                                                                 |


## Appendix: Detailed Evidence

### `Qwen/Qwen3-VL-2B-Instruct`

Observed signals:

- Output appears truncated to about 3 tokens.

Sample output:

```text
This image
```

### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

Observed signals:

- Output appears truncated to about 3 tokens.

Sample output:

```text
This image
```

### `mlx-community/Qwen3.5-35B-A3B-bf16`

Observed signals:

- Output appears truncated to about 3 tokens.

Sample output:

```text
os,
```

_Additional affected models are listed in the Affected Models table above._

