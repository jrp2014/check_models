<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm\]\[Stop-token leakage\] Stop/control tokens leaked into generated text affecting 2 model(s)

## Summary

2 model(s) show **Stop-token leakage** that should be filed against mlx-vlm.

- **Observed problem:** Stop/control tokens leaked into generated text
- **Target:** mlx-vlm
- **Affected models:** 2
- **Fixed when:** No leaked stop/control tokens.


## Model

- **Primary model:** `microsoft/Phi-3.5-vision-instruct`
- **Affected model count:** 2
- **Revision:** `unknown`
- **Trust remote code:** `true`


## Inputs

- **Prompt:** `Describe this image briefly.`
- **Image:** `/Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg`
- **Image facts:** 640x480 (0.31 MP), JPG extension, 173131 bytes, sha256 `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`
- **Shareable:** unknown


## Expected Behavior

- The native `mlx-vlm` CLI/Python repro should load the model, process the prompt and image, and return a response without the observed failure or quality regression.


## Actual Behavior

- Stop/control tokens leaked into generated text
- Representative signal: decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt;


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                           | Observed Behavior                                                                                                | Token Counts                                                                                       | Optional Context                                                                                                                                                                           |
|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `microsoft/Phi-3.5-vision-instruct`             | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; | prompt=770 \| output/prompt=25.97% \| nontext burden=99% \| stop=max_tokens \| hit token cap (200) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260614T214251Z_002_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)             |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | decoded text contains control token &lt;\|end\|&gt;                                                              | prompt=1,330 \| output/prompt=13.08% \| nontext burden=100% \| stop=completed                      | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260614T214251Z_003_mlx-community_Apriel-1.5-15b-Thinker-6bit-MLX_mlx_vlm_stop_token_001.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `microsoft/Phi-3.5-vision-instruct`: Special control token &lt;\|end\|&gt; appeared in generated text.
- `microsoft/Phi-3.5-vision-instruct`: Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output excerpt: `Two cats are sleeping on a pink couch with remote controls beside them.<\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|><\|endoftext\|><\|end\|...`
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Special control token &lt;\|end\|&gt; appeared in generated text.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model microsoft/Phi-3.5-vision-instruct --image /Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX --image /Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'microsoft/Phi-3.5-vision-instruct'
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
  "model": "microsoft/Phi-3.5-vision-instruct"
}
```

Optional advanced context:

- `microsoft/Phi-3.5-vision-instruct`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260614T214251Z_002_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260614T214251Z_003_mlx-community_Apriel-1.5-15b-Thinker-6bit-MLX_mlx_vlm_stop_token_001.json)
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
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,671,712 bytes, sha256=aea6719d5c77f5e26cd8c418c96a1358d4947f628fb4f56ca1ee252e3d504e89)  |
| RAM                        | 128.0 GB                                                                                                                                                 |


## Appendix: Detailed Evidence

### `microsoft/Phi-3.5-vision-instruct`

Observed signals:

- Special control token &lt;\|end\|&gt; appeared in generated text.
- Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

Sample output:

```text
Two cats are sleeping on a pink couch with remote controls beside them.<|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|...
```

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

Observed signals:

- Special control token &lt;\|end\|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).
- Output leaked reasoning or prompt-template text (here are my reasoning steps, the user asks:).

Sample output:

```text
Here are my reasoning steps:
The user asks: "Describe this image briefly." The image is of two cats sleeping on a pink couch, with a remote control. The user wants a brief description. This is a st...
```

