<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[model-config / mlx-vlm\]\[Prompt-template / image-placeholder mismatch\] Prompt/template output shape mismatch affecting 2 model(s)

## Summary

2 model(s) show **Prompt-template / image-placeholder mismatch** that should be filed against model repo first; mlx-vlm if template handling disagrees.

- **Observed problem:** Prompt/template output shape mismatch
- **Target:** model repo first; mlx-vlm if template handling disagrees
- **Affected models:** 2
- **Fixed when:** Requested sections render without template leakage.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                 | Observed Behavior   | Token Counts                                                                | Optional Context                                                                                                                                                                                   |
|---------------------------------------|---------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HuggingFaceTB/SmolVLM-Instruct`      | output/prompt=1.1%  | prompt=1,196 \| output/prompt=1.09% \| nontext burden=99% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260612T121808Z_001_HuggingFaceTB_SmolVLM-Instruct_model_config_mlx_vlm_prompt_template_001.json)      |
| `mlx-community/SmolVLM-Instruct-bf16` | output/prompt=1.1%  | prompt=1,196 \| output/prompt=1.09% \| nontext burden=99% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260612T121808Z_007_mlx-community_SmolVLM-Instruct-bf16_model_config_mlx_vlm_prompt_template_001.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `HuggingFaceTB/SmolVLM-Instruct`: Output is very short relative to prompt size (1.1%), suggesting possible early-stop or prompt-handling issues.
- Output excerpt: `Two cats are sleeping on a pink blanket on a couch.`
- `mlx-community/SmolVLM-Instruct-bf16`: Output is very short relative to prompt size (1.1%), suggesting possible early-stop or prompt-handling issues.
- Output excerpt: `Two cats are sleeping on a pink blanket on a couch.`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model HuggingFaceTB/SmolVLM-Instruct --image /Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/SmolVLM-Instruct-bf16 --image /Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'HuggingFaceTB/SmolVLM-Instruct'
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
  "model": "HuggingFaceTB/SmolVLM-Instruct"
}
```

Optional advanced context:

- `HuggingFaceTB/SmolVLM-Instruct`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260612T121808Z_001_HuggingFaceTB_SmolVLM-Instruct_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/SmolVLM-Instruct-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260612T121808Z_007_mlx-community_SmolVLM-Instruct-bf16_model_config_mlx_vlm_prompt_template_001.json)
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
| mlx-vlm                    | 0.6.3                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260612+337f736a                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
| transformers               | 5.11.0                                                                                                                                                   |
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
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,676,160 bytes, sha256=811f9557132c55cc5b95a4dcbdb6e3757ed88cfa5bcfabf8b3561959383335a5)  |
| RAM                        | 128.0 GB                                                                                                                                                 |


## Appendix: Detailed Evidence

### `HuggingFaceTB/SmolVLM-Instruct`

Observed signals:

- Output is very short relative to prompt size (1.1%), suggesting possible early-stop or prompt-handling issues.

Sample output:

```text
Two cats are sleeping on a pink blanket on a couch.
```

### `mlx-community/SmolVLM-Instruct-bf16`

Observed signals:

- Output is very short relative to prompt size (1.1%), suggesting possible early-stop or prompt-handling issues.

Sample output:

```text
Two cats are sleeping on a pink blanket on a couch.
```

