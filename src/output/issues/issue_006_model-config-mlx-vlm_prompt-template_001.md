<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[model-config / mlx-vlm\]\[Prompt-template / image-placeholder mismatch\] Prompt/template output shape mismatch affecting 3 model(s)

## Summary

3 model(s) show **Prompt-template / image-placeholder mismatch** that should be filed against model repo first; mlx-vlm if template handling disagrees.

- **Observed problem:** Prompt/template output shape mismatch
- **Target:** model repo first; mlx-vlm if template handling disagrees
- **Affected models:** 3
- **Fixed when:** Requested sections render without template leakage.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                      | Observed Behavior   | Token Counts                                                                 | Optional Context                                                                                                                                                                                        |
|--------------------------------------------|---------------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-3n-E2B-4bit`          | generated_tokens~1  | prompt=266 \| output/prompt=0.38% \| nontext burden=98% \| stop=completed    | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_010_mlx-community_gemma-3n-E2B-4bit_model_config_mlx_vlm_prompt_template_001.json)          |
| `mlx-community/gemma-4-31b-bf16`           | generated_tokens~1  | prompt=268 \| output/prompt=0.37% \| nontext burden=98% \| stop=completed    | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_011_mlx-community_gemma-4-31b-bf16_model_config_mlx_vlm_prompt_template_001.json)           |
| `mlx-community/llava-v1.6-mistral-7b-8bit` | output/prompt=0.6%  | prompt=2,160 \| output/prompt=0.56% \| nontext burden=100% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_012_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/gemma-3n-E2B-4bit`: Output appears truncated to about 1 tokens.
- `mlx-community/gemma-4-31b-bf16`: Output appears truncated to about 1 tokens.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/gemma-3n-E2B-4bit --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/gemma-4-31b-bf16 --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/llava-v1.6-mistral-7b-8bit --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/gemma-3n-E2B-4bit'
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
  "model": "mlx-community/gemma-3n-E2B-4bit"
}
```

Optional advanced context:

- `mlx-community/gemma-3n-E2B-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_010_mlx-community_gemma-3n-E2B-4bit_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/gemma-4-31b-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_011_mlx-community_gemma-4-31b-bf16_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/llava-v1.6-mistral-7b-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_012_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json)
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

### `mlx-community/gemma-3n-E2B-4bit`

Observed signals:

- Output appears truncated to about 1 tokens.

### `mlx-community/gemma-4-31b-bf16`

Observed signals:

- Output appears truncated to about 1 tokens.

### `mlx-community/llava-v1.6-mistral-7b-8bit`

Observed signals:

- Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues.

Sample output:

```text
A person riding a jet ski in the ocean.
```

