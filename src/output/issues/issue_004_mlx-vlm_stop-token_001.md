# \[mlx-vlm\]\[Stop-token leakage\] Stop/control tokens leaked into generated text affecting 1 model(s)

## Summary

1 model(s) show **Stop-token leakage** that should be filed against mlx-vlm.

- **Observed problem:** Stop/control tokens leaked into generated text
- **Target:** mlx-vlm
- **Affected models:** 1
- **Fixed when:** No leaked stop/control tokens.


## Affected Models

| Model                              | Observed Behavior                                         | Token Counts                                                                                       | Optional Context                                                                                                                                                              |
|------------------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/X-Reasoner-7B-8bit` | decoded text contains control token &lt;\|endoftext\|&gt; | prompt=803 \| output/prompt=24.91% \| nontext burden=99% \| stop=max_tokens \| hit token cap (200) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_009_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_stop_token_001.json) |


## Minimal Evidence

- `mlx-community/X-Reasoner-7B-8bit`: Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- `mlx-community/X-Reasoner-7B-8bit`: Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'of').
- Output excerpt: `Hamilton<\|endoftext\|> Image credit: 1.<\|endoftext\|> Image: A 1.<\|endoftext\|>Huge<\|endoftext\|>Image, 1.<\|endoftext\|>Newborns areal<\|endoftext\|>B<\|endoftext\|>B<\|endoftext\|>New 1.<\|endoftext\|>B<\|endoftext\|>Hilary is a woman.<\|endoftext\|>Beverly is a symbol of the city.<\|endoftext\|>B<\|endoftext\|>Hilary is a group of 20...`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/X-Reasoner-7B-8bit --image /Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --resize-shape 1024 1024 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/X-Reasoner-7B-8bit'
IMAGE = '/Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg'
PROMPT = 'Describe this image briefly.'
LOAD_KWARGS = {'trust_remote_code': True}
GENERATE_KWARGS = {'max_tokens': 200, 'temperature': 0.0, 'prefill_step_size': 4096, 'resize_shape': (1024, 1024)}
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
    "resize_shape": [
      1024,
      1024
    ],
    "temperature": 0.0
  },
  "image": "/Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/X-Reasoner-7B-8bit"
}
```

Optional advanced context:

- `mlx-community/X-Reasoner-7B-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_009_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_stop_token_001.json)
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
| mlx             | 0.32.0.dev20260516+7b7c1240 |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.8.1                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.15.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.5.0               |
| macOS Version   | 26.5                        |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |


## Appendix: Detailed Evidence

### `mlx-community/X-Reasoner-7B-8bit`

Observed signals:

- Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'of').
- Output switched language/script unexpectedly (tokenizer_artifact).

Sample output:

```text
Hamilton<|endoftext|> Image credit: 1.<|endoftext|> Image: A 1.<|endoftext|>Huge<|endoftext|>Image, 1.<|endoftext|>Newborns areal<|endoftext|>B<|endoftext|>B<|endoftext|>New 1.<|endoftext|>B<|endof...
```

