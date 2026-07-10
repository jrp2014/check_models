<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 3 model(s)

## Summary

3 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Affected models:** 3
- **Fixed when:** Full and reduced reruns avoid context collapse.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                     | Observed Behavior                                                                                                         | Token Counts                                                                 | Optional Context                                                                                                                                                                           |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Qwen/Qwen3-VL-2B-Instruct`               | generated_tokens~2 \| prompt_tokens=16729, output_tokens=2, output/prompt=0.0%, weak text=truncated                       | prompt=16,729 \| output/prompt=0.01% \| nontext burden=97% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_001_Qwen_Qwen3-VL-2B-Instruct_mlx_vlm_mlx_long_context_001.json)               |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16` | generated_tokens~2 \| prompt_tokens=16729, output_tokens=2, output/prompt=0.0%, weak text=truncated                       | prompt=16,729 \| output/prompt=0.01% \| nontext burden=97% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_003_mlx-community_Qwen3-VL-2B-Instruct-bf16_mlx_vlm_mlx_long_context_001.json) |
| `mlx-community/X-Reasoner-7B-8bit`        | output/prompt=0.1%, weak text=truncated \| prompt_tokens=16740, output_tokens=14, output/prompt=0.1%, weak text=truncated | prompt=16,740 \| output/prompt=0.08% \| nontext burden=97% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_006_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_001.json)        |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `Qwen/Qwen3-VL-2B-Instruct`: Output appears truncated to about 2 tokens.
- `Qwen/Qwen3-VL-2B-Instruct`: At long prompt length (16729 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated).
- Output excerpt: `-`
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: Output appears truncated to about 2 tokens.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.
Use a local copy of `20260704-181004_DSC00862_DxO.jpg` or replace it with an equivalent test image.
Image SHA256: `ca8d7f4e290d2f17ff550dd856e3cad8903013e2e0b5044bc926ee086199c806`

Native CLI:

```bash
python -m mlx_vlm.generate --model Qwen/Qwen3-VL-2B-Instruct --image 20260704-181004_DSC00862_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Title hint: , Deben Estuary, Woodbridge, England, UK, GBR, Europe
- Description hint: Two sailing boats moored on a river with trees behind on the bank
- Keyword hints: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England, Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging
- Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local time).' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3-VL-2B-Instruct-bf16 --image 20260704-181004_DSC00862_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Title hint: , Deben Estuary, Woodbridge, England, UK, GBR, Europe
- Description hint: Two sailing boats moored on a river with trees behind on the bank
- Keyword hints: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England, Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging
- Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local time).' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/X-Reasoner-7B-8bit --image 20260704-181004_DSC00862_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Title hint: , Deben Estuary, Woodbridge, England, UK, GBR, Europe
- Description hint: Two sailing boats moored on a river with trees behind on the bank
- Keyword hints: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England, Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging
- Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local time).' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'Qwen/Qwen3-VL-2B-Instruct'
IMAGE = '20260704-181004_DSC00862_DxO.jpg'
PROMPT = 'Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Existing metadata hints (high confidence; use only when visually confirmed):\n- Title hint: , Deben Estuary, Woodbridge, England, UK, GBR, Europe\n- Description hint: Two sailing boats moored on a river with trees behind on the bank\n- Keyword hints: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England, Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging\n- Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local time).'
LOAD_KWARGS = {'trust_remote_code': True}
GENERATE_KWARGS = {'max_tokens': 500, 'temperature': 0.0, 'prefill_step_size': 4096}
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
Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Title hint: , Deben Estuary, Woodbridge, England, UK, GBR, Europe
- Description hint: Two sailing boats moored on a river with trees behind on the bank
- Keyword hints: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England, Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging
- Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local time).
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 500,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "20260704-181004_DSC00862_DxO.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "Qwen/Qwen3-VL-2B-Instruct"
}
```

Optional advanced context:

- `Qwen/Qwen3-VL-2B-Instruct`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_001_Qwen_Qwen3-VL-2B-Instruct_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_003_mlx-community_Qwen3-VL-2B-Instruct-bf16_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/X-Reasoner-7B-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_006_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


## Appendix: Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.5                                                                                                                                                    |
| mlx                        | 0.32.1.dev20260710+4367c73b                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.5                                                                                                                                                    |
| transformers               | 5.13.0                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.23.0                                                                                                                                                   |
| Python Version             | 3.13.13                                                                                                                                                  |
| OS                         | Darwin 25.5.0                                                                                                                                            |
| macOS Version              | 26.5.2                                                                                                                                                   |
| SDK Version                | 26.5                                                                                                                                                     |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                                       |
| Xcode Version              | 26.6                                                                                                                                                     |
| Xcode Build                | 17F113                                                                                                                                                   |
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
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,449,848 bytes, sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=e61c827cd79f978aa5eacc136f65d6dea065005787f3a1457dc9d4512d6ee9cf)  |
| RAM                        | 128.0 GB                                                                                                                                                 |


## Appendix: Detailed Evidence

### `Qwen/Qwen3-VL-2B-Instruct`

Observed signals:

- Output appears truncated to about 2 tokens.
- At long prompt length (16729 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated).
- Model output may not follow prompt or image contents (missing: Bird, Boat, Boating, Buoy, Bushes).

Sample output:

```text
-
```

### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

Observed signals:

- Output appears truncated to about 2 tokens.
- At long prompt length (16729 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated).
- Model output may not follow prompt or image contents (missing: Bird, Boat, Boating, Buoy, Bushes).

Sample output:

```text
-
```

### `mlx-community/X-Reasoner-7B-8bit`

Observed signals:

- Output is very short relative to prompt size (0.1%) with weak text signal 'truncated', suggesting possible early-stop or prompt-handling issues.
- At long prompt length (16740 tokens), output stayed unusually short (14 tokens; ratio 0.1%; weak text signal truncated).
- Model output may not follow prompt or image contents (missing: Bird, Boat, Boating, Buoy, Bushes).
- Output contains corrupted or malformed text segments (encoding_shift).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
<|im_start|>
 addCriterion
要求：生成一个与图片内容相关的标题
```

