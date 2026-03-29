# Automated Review Digest

_Generated on 2026-03-29 00:43:26 GMT_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Canonical run log:_ [check_models.log](check_models.log)

## Maintainer Queue

### `model-config`

- `mlx-community/MolmoPoint-8B-fp16`: `harness` | processor_error, model_config_processor_load_processor | Inspect model repo config, chat template, and EOS settings.

## User Buckets

### `recommended`

- None.

### `caveat`

- None.

### `avoid`

- `mlx-community/MolmoPoint-8B-fp16`: `harness` | preserves trusted hints | processor_error, model_config_processor_load_processor

## Model Verdicts

### `mlx-community/MolmoPoint-8B-fp16`

- **Verdict:** harness | user=avoid
- **Why:** processor_error, model_config_processor_load_processor
- **Trusted hints:** not evaluated
- **Contract:** not evaluated
- **Utility:** user=avoid
- **Stack / owner:** owner=model-config | package=model-config | stage=Processor Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- **Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
- **Next action:** Inspect model repo config, chat template, and EOS settings.
