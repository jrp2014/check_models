# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 8 model(s)

## Summary

8 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Affected models:** 8
- **Fixed when:** Full and reduced reruns avoid context collapse.


## Affected Models

| Model                                     | Observed Behavior                                 | Token Counts                                                                                         | Optional Context                                                                                                                                                                           |
|-------------------------------------------|---------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | prompt_tokens=16851, repetitive output            | prompt=16,851 \| output/prompt=2.97% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_006_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json) |
| `mlx-community/Qwen3.5-27B-4bit`          | prompt_tokens=16866, repetitive output            | prompt=16,866 \| output/prompt=2.96% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_007_mlx-community_Qwen3.5-27B-4bit_mlx_vlm_mlx_long_context_001.json)          |
| `mlx-community/Qwen3.5-27B-mxfp8`         | prompt_tokens=16866, repetitive output            | prompt=16,866 \| output/prompt=2.96% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_008_mlx-community_Qwen3.5-27B-mxfp8_mlx_vlm_mlx_long_context_001.json)         |
| `mlx-community/Qwen3.5-35B-A3B-4bit`      | prompt_tokens=16866, repetitive output            | prompt=16,866 \| output/prompt=2.96% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_009_mlx-community_Qwen3.5-35B-A3B-4bit_mlx_vlm_mlx_long_context_001.json)      |
| `mlx-community/Qwen3.5-35B-A3B-6bit`      | prompt_tokens=16866, repetitive output            | prompt=16,866 \| output/prompt=2.96% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_010_mlx-community_Qwen3.5-35B-A3B-6bit_mlx_vlm_mlx_long_context_001.json)      |
| `mlx-community/Qwen3.5-35B-A3B-bf16`      | prompt_tokens=16866, repetitive output            | prompt=16,866 \| output/prompt=2.96% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_011_mlx-community_Qwen3.5-35B-A3B-bf16_mlx_vlm_mlx_long_context_001.json)      |
| `mlx-community/Qwen3.5-9B-MLX-4bit`       | prompt_tokens=16866, repetitive output            | prompt=16,866 \| output/prompt=2.96% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_012_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_001.json)       |
| `mlx-community/Qwen3.6-27B-mxfp8`         | prompt_tokens=16866, prompt/image context dropped | prompt=16,866 \| output/prompt=2.96% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_013_mlx-community_Qwen3.6-27B-mxfp8_mlx_vlm_mlx_long_context_001.json)         |


## Minimal Evidence

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: At long prompt length (16851 tokens), output became repetitive.
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: Model output may not follow prompt or image contents (missing: Architecture, Bench, Bird, Building, Bush).
- Output excerpt: `The video, and I am. The digital and digital, and the camera. and 100, and a digital, and a large, and a bunch, and a lot, and a lot of them, 100, and 100, and a large, and a bunch, they are, and wooden, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, it is, and a,...`
- `mlx-community/Qwen3.5-27B-4bit`: At long prompt length (16866 tokens), output became repetitive.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The tall spire of St John the Evangelist'"'"'s Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-27B-4bit --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The tall spire of St John the Evangelist'"'"'s Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-27B-mxfp8 --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The tall spire of St John the Evangelist'"'"'s Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-35B-A3B-4bit --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The tall spire of St John the Evangelist'"'"'s Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-35B-A3B-6bit --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The tall spire of St John the Evangelist'"'"'s Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-35B-A3B-bf16 --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The tall spire of St John the Evangelist'"'"'s Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The tall spire of St John the Evangelist'"'"'s Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.6-27B-mxfp8 --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The tall spire of St John the Evangelist'"'"'s Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/Qwen2-VL-2B-Instruct-4bit'
IMAGE = '/Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg'
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Existing metadata hints (high confidence; use only when visually confirmed):\n- Description hint: The tall spire of St John the Evangelist's Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.\n- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors\n- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W."
LOAD_KWARGS = {'trust_remote_code': True}
GENERATE_KWARGS = {'max_tokens': 500, 'temperature': 0.0, 'prefill_step_size': 4096}
model, processor = load(MODEL, **LOAD_KWARGS)
result = generate(model, processor, PROMPT, image=IMAGE, **GENERATE_KWARGS)
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
- Description hint: The tall spire of St John the Evangelist's Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 500,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "/Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit"
}
```

Optional advanced context:

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_006_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/Qwen3.5-27B-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_007_mlx-community_Qwen3.5-27B-4bit_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/Qwen3.5-27B-mxfp8`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_008_mlx-community_Qwen3.5-27B-mxfp8_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/Qwen3.5-35B-A3B-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_009_mlx-community_Qwen3.5-35B-A3B-4bit_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/Qwen3.5-35B-A3B-6bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_010_mlx-community_Qwen3.5-35B-A3B-6bit_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/Qwen3.5-35B-A3B-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_011_mlx-community_Qwen3.5-35B-A3B-bf16_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/Qwen3.5-9B-MLX-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_012_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/Qwen3.6-27B-mxfp8`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_013_mlx-community_Qwen3.6-27B-mxfp8_mlx_vlm_mlx_long_context_001.json)
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
| mlx             | 0.32.0.dev20260515+7b7c1240 |
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

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

Observed signals:

- At long prompt length (16851 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: Architecture, Bench, Bird, Building, Bush).
- Output became repetitive, indicating possible generation instability (token: phrase: "and a, and a,...").
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
The video, and I am. The digital and digital, and the camera.  and 100, and a digital, and a large, and a bunch, and a lot, and a lot of them, 100, and 100, and a large, and a bunch, they are, and...
```

### `mlx-community/Qwen3.5-27B-4bit`

Observed signals:

- At long prompt length (16866 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: Architecture, Bench, Bird, Building, Bush).
- Output became repetitive, indicating possible generation instability (token: 1).
- Output contains corrupted or malformed text segments (character_loop: '1 ' repeated).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
adata竞人满大值走 下 稍 信 息 和 5 6 7 8 9 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1...
```

### `mlx-community/Qwen3.5-27B-mxfp8`

Observed signals:

- At long prompt length (16866 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: Architecture, Bench, Bird, Building, Bush).
- Output became repetitive, indicating possible generation instability (token: phrase: "- 伊 - 德...").
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
μάτων维奇喧

afd.
- 第一
- 弗
- 洛
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
- 伊
- 德
- 罗
-...
```

_Additional affected models are listed in the Affected Models table above._

