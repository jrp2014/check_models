# Diagnostics Report — 1 failure(s), 0 harness issue(s) (mlx-vlm 0.3.11)

## Summary

Automated benchmarking of **1 locally-cached VLM models** found **1 hard failure(s)** and **0 harness/integration issue(s)** in successful models. 0 of 1 models succeeded.

Test image: `e2e_test.jpg` (0.0 MB). All failures are deterministic and reproducible.

## Environment

| Component | Version |
|-----------|---------|
| mlx-vlm | 0.3.11 |
| mlx | 0.30.7.dev20260207+8fe1d092 |
| mlx-lm | 0.30.6 |
| transformers | 5.1.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.4.1 |
| Python Version | 3.13.9 |
| OS | Darwin 25.2.0 |
| macOS Version | 26.2 |
| GPU/Chip | Apple M4 Max |
| GPU Cores | 40 |
| Metal Support | Metal 4 |
| RAM | 128.0 GB |

---

## 1. Model Error — 1 model(s) [`mlx-vlm`] (Priority: Medium)

**Error:** `Model loading failed: 404 Client Error. (Request ID: Root=1-6987cc92-791d6a231660815560780f60;65c7b8a1-86d7-419e-98c2-37ed26c127db)`

| Model | Error Stage | Package |
|-------|-------------|---------|
| `nonexistent/fake-model-12345` | Model Error | mlx-vlm |

**Traceback (tail):**

```
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6863, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: 404 Client Error. (Request ID: Root=1-6987cc92-791d6a231660815560780f60;65c7b8a1-86d7-419e-98c2-37ed26c127db)
Repository Not Found for url: https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication
```

---

## Priority Summary

| Priority | Issue | Models Affected | Package |
|----------|-------|-----------------|---------|
| **Medium** | Model Error | 1 (fake-model-12345) | mlx-vlm |

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e src/.[dev]

# Run against all cached models
python src/check_models.py

# Target specific failing models:
python src/check_models.py --model nonexistent/fake-model-12345
```

<details><summary>Prompt used (click to expand)</summary>

```
Analyze this image and provide structured metadata for image cataloguing.

Respond with exactly these three sections:

Title: A concise, descriptive title (5–12 words).

Description: A factual 1–3 sentence description of what the image shows, covering key subjects, setting, and action.

Keywords: 25–50 comma-separated keywords ordered from most specific to most general, covering:
- Specific subjects (people, objects, animals, landmarks)
- Setting and location (urban, rural, indoor, outdoor)
- Actions and activities
- Concepts and themes (e.g. teamwork, solitude, celebration)
- Mood and emotion (e.g. serene, dramatic, joyful)
- Visual style (e.g. close-up, wide-angle, aerial, silhouette, bokeh)
- Colors and lighting (e.g. golden hour, blue tones, high-key)
- Seasonal and temporal context
- Use-case relevance (e.g. business, travel, food, editorial)
Taken around 2026-02-07 23:36:50 GMT at 23:36:50.

Be factual about visual content.  Include relevant conceptual and emotional keywords where clearly supported by the image.
```

</details>

_Report generated on 2026-02-07 23:37:20 GMT by [check_models](https://github.com/jrp2014/check_models)._