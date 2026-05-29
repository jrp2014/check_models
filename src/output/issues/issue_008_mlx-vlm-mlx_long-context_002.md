<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 2 model(s)

## Summary

2 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Affected models:** 2
- **Fixed when:** Full and reduced reruns avoid context collapse.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                | Observed Behavior                                                             | Token Counts                                                                                          | Optional Context                                                                                                                                                                      |
|--------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.5-35B-A3B-4bit` | Output degeneration under long prompt length (repeated_punctuation: ':**...') | prompt=16,241 \| output/prompt=1.23% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260529T205941Z_010_mlx-community_Qwen3.5-35B-A3B-4bit_mlx_vlm_mlx_long_context_002.json) |
| `mlx-community/Qwen3.5-9B-MLX-4bit`  | Output degeneration under long prompt length (repeated_punctuation: ':**...') | prompt=16,241 \| output/prompt=1.23% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260529T205941Z_011_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_002.json)  |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/Qwen3.5-35B-A3B-4bit`: Output degeneration under long prompt length (repeated_punctuation: ':**...')
- Output excerpt: `The user wants a brief description of the image. 1. **Identify the main subject:** A young boy riding a jet ski. 2. **Describe the action:** He is riding on water, creating a wake. He is looking at the camera and smiling. He has one hand on the handlebar and the other raised in a fist pump gesture. 3. **Describe...`
- `mlx-community/Qwen3.5-9B-MLX-4bit`: Output degeneration under long prompt length (repeated_punctuation: ':**...')
- Output excerpt: `The user wants a brief description of the image. 1. **Identify the main subject:** A young boy riding a jet ski. 2. **Identify the action:** He is riding fast, creating a wake, and looking back at the camera while raising his fist in a "rock on" or victory gesture. 3. **Identify the setting:** Greenish water, li...`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-35B-A3B-4bit --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/Qwen3.5-35B-A3B-4bit'
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
  "model": "mlx-community/Qwen3.5-35B-A3B-4bit"
}
```

Optional advanced context:

- `mlx-community/Qwen3.5-35B-A3B-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260529T205941Z_010_mlx-community_Qwen3.5-35B-A3B-4bit_mlx_vlm_mlx_long_context_002.json)
- `mlx-community/Qwen3.5-9B-MLX-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260529T205941Z_011_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_002.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


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

### Stack Signals

| Model                                |   Prompt Tok |   Output Tok | Output/Prompt   | Symptom                                                                       | Owner           |
|--------------------------------------|--------------|--------------|-----------------|-------------------------------------------------------------------------------|-----------------|
| `mlx-community/Qwen3.5-35B-A3B-4bit` |       16,241 |          200 | 1.23%           | Output degeneration under long prompt length (repeated_punctuation: ':**...') | `mlx-vlm / mlx` |
| `mlx-community/Qwen3.5-9B-MLX-4bit`  |       16,241 |          200 | 1.23%           | Output degeneration under long prompt length (repeated_punctuation: ':**...') | `mlx-vlm / mlx` |

### `mlx-community/Qwen3.5-35B-A3B-4bit`

Observed signals:

- Output contains corrupted or malformed text segments (repeated_punctuation: ':**...').

Sample output:

```text
The user wants a brief description of the image.

1.  **Identify the main subject:** A young boy riding a jet ski.
2.  **Describe the action:** He is riding on water, creating a wake. He is looking...
```

### `mlx-community/Qwen3.5-9B-MLX-4bit`

Observed signals:

- Output contains corrupted or malformed text segments (repeated_punctuation: ':**...').

Sample output:

```text
The user wants a brief description of the image.

1.  **Identify the main subject:** A young boy riding a jet ski.
2.  **Identify the action:** He is riding fast, creating a wake, and looking back...
```

