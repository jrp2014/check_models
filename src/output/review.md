# Automated Review Digest

_Generated on 2026-03-28 23:42:42 GMT_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [../../../../../../../../private/var/folders/tk/zd2fp8lj21l9h1l1hrs7pq800000gn/T/pytest-of-jrp/pytest-43/test_invalid_model_produces_er0/output/e2e.log](../../../../../../../../private/var/folders/tk/zd2fp8lj21l9h1l1hrs7pq800000gn/T/pytest-of-jrp/pytest-43/test_invalid_model_produces_er0/output/e2e.log)

## Maintainer Queue

### `model`

- `nonexistent/fake-model-12345`: `harness` | model_error, huggingface_hub_model_load_model | Treat as a model-quality limitation for this prompt and image.

## User Buckets

### `recommended`

- None.

### `caveat`

- None.

### `avoid`

- `nonexistent/fake-model-12345`: `harness` | preserves trusted hints | model_error, huggingface_hub_model_load_model

## Model Verdicts

### `nonexistent/fake-model-12345`

- **Verdict:** harness | user=avoid
- **Why:** model_error, huggingface_hub_model_load_model
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=model | package=huggingface-hub | stage=Model Error | code=HUGGINGFACE_HUB_MODEL_LOAD_MODEL
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Treat as a model-quality limitation for this prompt and image.
