# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 2 model(s)

## Summary

2 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Affected models:** 2
- **Fixed when:** Full and reduced reruns avoid context collapse.


## Affected Models

| Model                             | Observed Behavior                                                            | Token Counts                                                                                          | Optional Context                                                                                                                                                                   |
|-----------------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.5-27B-4bit`  | Output degeneration under long prompt length (character_loop: '00' repeated) | prompt=16,167 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260517T213817Z_009_mlx-community_Qwen3.5-27B-4bit_mlx_vlm_mlx_long_context_003.json)  |
| `mlx-community/Qwen3.6-27B-mxfp8` | Output degeneration under long prompt length (character_loop: '66' repeated) | prompt=16,167 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260517T213817Z_014_mlx-community_Qwen3.6-27B-mxfp8_mlx_vlm_mlx_long_context_003.json) |


## Minimal Evidence

- `mlx-community/Qwen3.5-27B-4bit`: Output degeneration under long prompt length (character_loop: '00' repeated)
- Output excerpt: `猛进arpakh,2000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000`
- `mlx-community/Qwen3.6-27B-mxfp8`: Output degeneration under long prompt length (character_loop: '66' repeated)
- Output excerpt: `深夜atraicher Mala6666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-27B-4bit --image /Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.6-27B-mxfp8 --image /Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/Qwen3.5-27B-4bit'
IMAGE = '/Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg'
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
  "image": "/Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/Qwen3.5-27B-4bit"
}
```

Optional advanced context:

- `mlx-community/Qwen3.5-27B-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260517T213817Z_009_mlx-community_Qwen3.5-27B-4bit_mlx_vlm_mlx_long_context_003.json)
- `mlx-community/Qwen3.6-27B-mxfp8`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260517T213817Z_014_mlx-community_Qwen3.6-27B-mxfp8_mlx_vlm_mlx_long_context_003.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


## Appendix: Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260517+7b7c1240 |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.8.1                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.15.0                      |
| Python Version  | 3.13.13                     |
| OS              | Darwin 25.5.0               |
| macOS Version   | 26.5                        |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |


## Appendix: Detailed Evidence

### Stack Signals

| Model                             |   Prompt Tok |   Output Tok | Output/Prompt   | Symptom                                                                      | Owner           |
|-----------------------------------|--------------|--------------|-----------------|------------------------------------------------------------------------------|-----------------|
| `mlx-community/Qwen3.5-27B-4bit`  |       16,167 |          200 | 1.24%           | Output degeneration under long prompt length (character_loop: '00' repeated) | `mlx-vlm / mlx` |
| `mlx-community/Qwen3.6-27B-mxfp8` |       16,167 |          200 | 1.24%           | Output degeneration under long prompt length (character_loop: '66' repeated) | `mlx-vlm / mlx` |

### `mlx-community/Qwen3.5-27B-4bit`

Observed signals:

- Output contains corrupted or malformed text segments (character_loop: '00' repeated).

Sample output:

```text
猛进arpakh,20000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000...
```

### `mlx-community/Qwen3.6-27B-mxfp8`

Observed signals:

- Output contains corrupted or malformed text segments (character_loop: '66' repeated).

Sample output:

```text
深夜atraicher Mala6666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666...
```

