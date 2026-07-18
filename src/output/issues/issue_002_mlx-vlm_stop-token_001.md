<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm\]\[Stop-token leakage\] Stop/control tokens leaked into generated text affecting 1 model(s)

## Summary

1 model(s) show **Stop-token leakage** that should be filed against mlx-vlm.

- **Observed problem:** Stop/control tokens leaked into generated text
- **Target:** mlx-vlm
- **Affected models:** 1
- **Fixed when:** No leaked stop/control tokens.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                     | Observed Behavior                                  | Token Counts                                         | Optional Context                                                                                                                                                                     |
|-------------------------------------------|----------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | decoded text contains control token &lt;/think&gt; | prompt=317 \| output/prompt=59.31% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_008_mlx-community_Qwen3-VL-2B-Thinking-bf16_mlx_vlm_stop_token_001.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: Special control token &lt;/think&gt; appeared in generated text.
- Output excerpt: `...blue button. The background is the pink couch, so the main elements are the two cats, the remotes, and the pink fabric. I need to describe this briefly. &lt;/think&gt; Two tabby cats are lying on a bright pink couch, relaxed and asleep. A white remote control with a blue button sits between them, and another remote co...`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.
Use a local copy of `cats.jpg` or replace it with an equivalent test image.
Image SHA256: `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`

## Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3-VL-2B-Thinking-bf16 --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'mlx-community/Qwen3-VL-2B-Thinking-bf16'
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
  "model": "mlx-community/Qwen3-VL-2B-Thinking-bf16"
}
```

Optional advanced context:

- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260718T182321Z_008_mlx-community_Qwen3-VL-2B-Thinking-bf16_mlx_vlm_stop_token_001.json)
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

| Component                  | Version                                                                                                                                         |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.5                                                                                                                                           |
| mlx                        | 0.32.1.dev20260718+b7c3dd6d                                                                                                                     |
| mlx-lm                     | 0.31.3                                                                                                                                          |
| mlx-audio                  | 0.4.5                                                                                                                                           |
| transformers               | 5.14.1                                                                                                                                          |
| tokenizers                 | 0.22.2                                                                                                                                          |
| huggingface-hub            | 1.24.0                                                                                                                                          |
| Python Version             | 3.13.13                                                                                                                                         |
| OS                         | Darwin 25.5.0                                                                                                                                   |
| macOS Version              | 26.5.2                                                                                                                                          |
| SDK Version                | 26.5                                                                                                                                            |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                              |
| Xcode Version              | 26.6                                                                                                                                            |
| Xcode Build                | 17F113                                                                                                                                          |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                      |
| Metal SDK                  | MacOSX26.5.sdk                                                                                                                                  |
| Metal Compiler Version     | Apple metal version 32023.883 (metalfe-32023.883)                                                                                               |
| Metallib Linker Version    | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                  |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                 |
| GPU/Chip                   | Apple M5 Max                                                                                                                                    |
| GPU Cores                  | 40                                                                                                                                              |
| Metal Support              | Metal 4                                                                                                                                         |
| MLX Install Type           | editable local source                                                                                                                           |
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,449,848 bytes, sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=c03f8b1be065c164ed6c9380cc87b205f1c11c0e5b0cda028984159c57c79a3b)  |
| RAM                        | 128.0 GB                                                                                                                                        |


## Appendix: Detailed Evidence

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

Observed signals:

- Special control token &lt;/think&gt; appeared in generated text.

Sample output:

```text
...lements are the two cats, the remotes, and the pink fabric. I need to describe this briefly.
</think>

Two tabby cats are lying on a bright pink couch, relaxed and asleep. A white remote control...
```

