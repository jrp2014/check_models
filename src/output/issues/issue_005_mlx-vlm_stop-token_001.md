# \[mlx-vlm\]\[Stop-token leakage\] Stop/control tokens leaked into generated text affecting 2 model(s)

## Summary

2 model(s) show **Stop-token leakage** that should be filed against mlx-vlm.

- **Observed problem:** Stop/control tokens leaked into generated text
- **Target:** mlx-vlm
- **Affected models:** 2
- **Fixed when:** No leaked stop/control tokens.


## Affected Models

| Model                                     | Observed Behavior                                         | Token Counts                                                                                          | Optional Context                                                                                                           |
|-------------------------------------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | decoded text contains control token &lt;\|endoftext\|&gt; | prompt=16,176 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](../repro_bundles/20260521T221248Z_008_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_stop_token_001.json) |
| `mlx-community/X-Reasoner-7B-8bit`        | decoded text contains control token &lt;\|endoftext\|&gt; | prompt=16,176 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](../repro_bundles/20260521T221248Z_015_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_stop_token_001.json)        |


## Minimal Evidence

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: Output switched language/script unexpectedly (tokenizer_artifact).
- Output excerpt: `. The Blueprints and 10. We are in and in the United States, I and my friends, and find the door. The sky is a bit, and the battery, and I, and I. It is the fault, and I am a, and I. The sky is blue and a, the sky. The sky is not a new, and I with me and a, and I with me, it is, and I and my friends, and they are in...`
- `mlx-community/X-Reasoner-7B-8bit`: Special control token &lt;\|endoftext\|&gt; appeared in generated text.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --image /Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/X-Reasoner-7B-8bit --image /Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/Qwen2-VL-2B-Instruct-4bit'
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
  "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit"
}
```

Optional advanced context:

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: [optional JSON](../repro_bundles/20260521T221248Z_008_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_stop_token_001.json)
- `mlx-community/X-Reasoner-7B-8bit`: [optional JSON](../repro_bundles/20260521T221248Z_015_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_stop_token_001.json)
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

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260521+5d1c0e4c |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.9.0                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.16.1                      |
| Python Version  | 3.13.13                     |
| OS              | Darwin 25.5.0               |
| macOS Version   | 26.5                        |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |


## Appendix: Detailed Evidence

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

Observed signals:

- Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

Sample output:

```text
. The Blueprints and 10. We are in and in the United States, I and my friends, and find the door. The sky is a bit, and the battery, and I, and I. It is the fault, and I am a, and I. The sky is blu...
```

### `mlx-community/X-Reasoner-7B-8bit`

Observed signals:

- Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

Sample output:

```text
B<|endoftext|>, 1.<|endoftext|>-100<|endoftext|>-<|endoftext|>1<|endoftext|>-<|endoftext|>-<|endoftext|>-<|endoftext|>- 1. The 201.<|endoftext|>The 2010 2010 - 2008<|endoftext|>B<|endoftext|>-<|end...
```

