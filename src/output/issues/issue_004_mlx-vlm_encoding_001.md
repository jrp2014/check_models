<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm\]\[Tokenizer / decoding artifact\] Tokenizer decode leaked BPE/byte markers affecting 1 model(s)

## Summary

1 model(s) show **Tokenizer / decoding artifact** that should be filed against mlx-vlm.

- **Observed problem:** Tokenizer decode leaked BPE/byte markers
- **Target:** mlx-vlm
- **Affected models:** 1
- **Fixed when:** No BPE/byte markers in output.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                                   | Observed Behavior                           | Token Counts                                                                 | Optional Context                                                                                                                                                                                 |
|---------------------------------------------------------|---------------------------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | 126 BPE space markers found in decoded text | prompt=1,745 \| output/prompt=9.28% \| nontext burden=100% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 126 occurrences).
- Output excerpt: `TheĠimageĠdepictsĠtheĠentranceĠtoĠaĠbrewery,ĠspecificallyĠMolsonĠCoors,ĠasĠindicatedĠbyĠtheĠsignage.ĠTheĠbreweryĠisĠcharacterizedĠbyĠlarge,ĠcylindricalĠstorageĠtanks,ĠwhichĠareĠtypicalĠforĠbeerĠproduction.ĠTheĠentranceĠisĠmarkedĠbyĠaĠpairĠofĠstoneĠpillarsĠwithĠdecorativeĠfinials,ĠandĠaĠblackĠwrought-ironĠgate.ĠTheĠs...`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit --image /Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit'
IMAGE = '/Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg'
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
  "image": "/Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit"
}
```

Optional advanced context:

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns contain no leaked BPE, byte-level, or tokenizer marker text.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect tokenizer decode cleanup for byte-level/BPE marker leakage.
- [ ] Compare `decode` and `batch_decode` behavior with `skip_special_tokens=True`.
- [ ] Verify processor/tokenizer config does not require model-specific cleanup flags.


## Appendix: Environment

| Component       | Version       |
|-----------------|---------------|
| mlx-vlm         | 0.5.0         |
| mlx             | 0.31.2        |
| mlx-lm          | 0.31.3        |
| mlx-audio       | 0.4.3         |
| transformers    | 5.9.0         |
| tokenizers      | 0.22.2        |
| huggingface-hub | 1.16.1        |
| Python Version  | 3.13.13       |
| OS              | Darwin 25.5.0 |
| macOS Version   | 26.5          |
| GPU/Chip        | Apple M5 Max  |
| GPU Cores       | 40            |
| Metal Support   | Metal 4       |
| RAM             | 128.0 GB      |


## Appendix: Detailed Evidence

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

Observed signals:

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 126 occurrences).

Sample output:

```text
TheĠimageĠdepictsĠtheĠentranceĠtoĠaĠbrewery,ĠspecificallyĠMolsonĠCoors,ĠasĠindicatedĠbyĠtheĠsignage.ĠTheĠbreweryĠisĠcharacterizedĠbyĠlarge,ĠcylindricalĠstorageĠtanks,ĠwhichĠareĠtypicalĠforĠbeerĠpro...
```

