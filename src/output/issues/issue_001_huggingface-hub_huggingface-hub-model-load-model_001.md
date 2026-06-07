<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[huggingface_hub\]\[Hugging Face Hub: Model load / model error\] Hugging Face Hub: Model load / model error: [Errno 2] No such file or directory: '/Users/jrp/ affecting 1 model(s)

## Summary

1 model(s) show **Hugging Face Hub: Model load / model error** that should be filed against huggingface_hub.

- **Observed problem:** Hugging Face Hub: Model load / model error: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook-
- **Target:** huggingface_hub
- **Affected models:** 1
- **Fixed when:** Load/generation completes or fails with a narrower owner.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                  | Observed Behavior                                                                                                                                                                                                                                                                                                                             | Token Counts   | Optional Context                                                                                                                                                                    |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `facebook/pe-av-large` | [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete' -> '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354' | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260607T203551Z_003_facebook_pe-av-large_HUGGINGFACE_HUB_MODEL_LOAD_MODEL_d6788a6.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `facebook/pe-av-large` fails with: Model loading failed: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete'
- Root exception: `builtins.FileNotFoundError`: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d7...


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model facebook/pe-av-large --image /Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'facebook/pe-av-large'
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
  "model": "facebook/pe-av-large"
}
```

Optional advanced context:

- `facebook/pe-av-large`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260607T203551Z_003_facebook_pe-av-large_HUGGINGFACE_HUB_MODEL_LOAD_MODEL_d6788a6.json)
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
| mlx-vlm                    | 0.6.2                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260607+8f0e8b14                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
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
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (157,751,704 bytes, sha256=ba9913d81d92bbbde42bbc6dda27e80ecb31db6031fa073e6c8aeb0666d47c33) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,675,136 bytes, sha256=6255fc531acc826e8625f261237f6bb6c75490177d3b769ab70c1ff9f71b6d7f)  |
| RAM                        | 128.0 GB                                                                                                                                                 |


## Appendix: Detailed Evidence

### `facebook/pe-av-large`

Observed error:

```text
Model loading failed: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete'
```

Root exception:

```text
builtins.FileNotFoundError: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete' -> '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354'
```

Traceback tail:

```text
    with open(src, 'rb') as fsrc:
         ~~~~^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete'
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete'
```

