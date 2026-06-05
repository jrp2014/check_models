<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm\]\[Stop-token leakage\] Stop/control tokens leaked into generated text affecting 4 model(s)

## Summary

4 model(s) show **Stop-token leakage** that should be filed against mlx-vlm.

- **Observed problem:** Stop/control tokens leaked into generated text
- **Target:** mlx-vlm
- **Affected models:** 4
- **Fixed when:** No leaked stop/control tokens.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                | Observed Behavior                                                                                                | Token Counts                                                                                         | Optional Context                                                                                                                                                                |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `microsoft/Phi-3.5-vision-instruct`  | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; | prompt=770 \| output/prompt=25.97% \| nontext burden=99% \| stop=max_tokens \| hit token cap (200)   | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_002_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)  |
| `mlx-community/GLM-4.6V-Flash-6bit`  | decoded text contains control token &lt;/think&gt;                                                               | prompt=6,157 \| output/prompt=2.96% \| nontext burden=100% \| stop=completed                         | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_004_mlx-community_GLM-4.6V-Flash-6bit_mlx_vlm_stop_token_001.json)  |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | decoded text contains control token &lt;/think&gt;                                                               | prompt=6,157 \| output/prompt=3.25% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_005_mlx-community_GLM-4.6V-Flash-mxfp4_mlx_vlm_stop_token_001.json) |
| `mlx-community/GLM-4.6V-nvfp4`       | decoded text contains control token &lt;/think&gt;                                                               | prompt=6,157 \| output/prompt=3.25% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_006_mlx-community_GLM-4.6V-nvfp4_mlx_vlm_stop_token_001.json)       |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `microsoft/Phi-3.5-vision-instruct`: Special control token &lt;\|end\|&gt; appeared in generated text.
- `microsoft/Phi-3.5-vision-instruct`: Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output excerpt: `The image captures a dynamic scene of a person riding a green and black jet ski on a body of water. The rider, wearing a black life jacket and gloves, is in motion, with the jet ski creating a spray of water behind it. The water is a light green color, and the jet ski is marked with the text 'AG-8933'. The rider's p...`
- `mlx-community/GLM-4.6V-Flash-6bit`: Special control token &lt;/think&gt; appeared in generated text.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model microsoft/Phi-3.5-vision-instruct --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/GLM-4.6V-Flash-6bit --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/GLM-4.6V-Flash-mxfp4 --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/GLM-4.6V-nvfp4 --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'microsoft/Phi-3.5-vision-instruct'
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
  "model": "microsoft/Phi-3.5-vision-instruct"
}
```

Optional advanced context:

- `microsoft/Phi-3.5-vision-instruct`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_002_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)
- `mlx-community/GLM-4.6V-Flash-6bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_004_mlx-community_GLM-4.6V-Flash-6bit_mlx_vlm_stop_token_001.json)
- `mlx-community/GLM-4.6V-Flash-mxfp4`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_005_mlx-community_GLM-4.6V-Flash-mxfp4_mlx_vlm_stop_token_001.json)
- `mlx-community/GLM-4.6V-nvfp4`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T123125Z_006_mlx-community_GLM-4.6V-nvfp4_mlx_vlm_stop_token_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect model EOS token IDs and tokenizer special-token mappings.
- [ ] Verify mlx-vlm stop criteria receive all configured EOS/stop tokens.
- [ ] Check `skip_special_tokens` handling during decode.
- [ ] Strip generated control tokens such as `<|end|>` and `</think>` only after confirming generation stopped at the right boundary.


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

### `microsoft/Phi-3.5-vision-instruct`

Observed signals:

- Special control token &lt;\|end\|&gt; appeared in generated text.
- Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact, code_snippet).

Sample output:

```text
The image captures a dynamic scene of a person riding a green and black jet ski on a body of water. The rider, wearing a black life jacket and gloves, is in motion, with the jet ski creating a spra...
```

### `mlx-community/GLM-4.6V-Flash-6bit`

Observed signals:

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

Sample output:

```text
<think>Got it, let's see. The image shows a person riding a Sea-Doo jet ski in the ocean. The jet ski is green and yellow with the registration AG-8933. The rider is wearing a life vest, black shir...
```

### `mlx-community/GLM-4.6V-Flash-mxfp4`

Observed signals:

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

Sample output:

```text
<think>Got it, let's see. The image shows a person riding a green and yellow Sea-Doo jet ski on the water. The jet ski has the registration "AG-8933". The person is wearing a black life vest, black...
```

_Additional affected models are listed in the Affected Models table above._

