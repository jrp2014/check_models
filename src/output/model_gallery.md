# Model Output Gallery

_Generated on 2026-03-08 22:49:14 GMT_

A review-friendly artifact with image metadata, the source prompt, and full generated output for each model.

## Image Metadata

- **Date**: 2026-03-08 22:48:58 GMT
- **Time**: 22:48:58

## Prompt

```text
Analyze this image for cataloguing metadata.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title: 5-10 words, concrete and factual, limited to clearly visible content.

Description: 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.

Keywords: 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Capture metadata hints: Taken on 2026-03-08 22:48:58 GMT (at 22:48:58 local time).
```

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD013 MD033 -->

### ✅ mlx-community/FastVLM-0.5B-bf16

**Metrics:** 314 TPS | 120 tokens

```text
Title: 5-10 words, concrete and factual, limited to clearly visible content.

Description: The image depicts a simple, stylized representation of a house and a sun against a light blue background. The house is a brown rectangular structure with a red triangular roof, and it is situated on a green rectangular base. The sun is a yellow circle in the upper right corner of the image. The image is clear and does not contain any ambiguous or uncertain details. The metadata suggests that the image was taken on March 8, 2026, at 22:48
```

---

<!-- markdownlint-enable MD013 MD033 -->
