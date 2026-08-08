---
name: hf-cache-mlx-vlm-models
description: >
  List or reason about local Hugging Face cache models that match the mlx-vlm
  server-supported discovery filter used by check_models default model
  selection. Use for cache-dir questions, skipped-repo reasons, dry-run model
  lists, or aligning discovery with /v1/models. This is a file-presence check,
  not a generation proof. This repo uses conda + pip, never uv.
---

# HF Cache Models (mlx-vlm server filter)

Default `check_models` discovery intentionally matches the **mlx-vlm server
`/v1/models` cache filter**, not “every repo in the HF cache”.

Adapted from upstream mlx-vlm support skills
([Blaizzy/mlx-vlm#1343](https://github.com/Blaizzy/mlx-vlm/pull/1343)).

## Supported-model rule

A cached repo is treated as server-supported when **all** of the following hold:

- repo type is `model`
- a `main` revision exists in the cache
- `config.json` is present on that revision
- `tokenizer_config.json` is present
- weights exist as `model.safetensors.index.json` **or** at least one
  `*.safetensors` file

This is a **cache/file-presence** check. It does not load the model or prove
generation works.

## Architecture pre-check (upstream `--check-arch` tier)

`check_models` also mirrors the upstream `hf-cache-models --check-arch` tier:
each cached repo's `config.json` `model_type` (falling back to
`speculators_model_type`, lowercased, with `MODEL_REMAPPING` aliases parsed
from the installed mlx-vlm source — never importing mlx) is compared against
the package directories under the installed `mlx_vlm/models/`.

- `--dry-run` annotates models whose architecture is not supported and prints
  an unsupported-architecture count.
- The per-model gallery/diagnostics fact "Arch supported by installed
  mlx-vlm" and the optional `architecture` record in `results.jsonl` carry the
  same verdict (`model_type`, `resolved_model_type`,
  `supported_by_installed_mlx_vlm`).
- This is a **folder-name check only**: "unsupported" means "probably not
  loadable", never proof. Unsupported models are still attempted so real
  crash evidence is captured; a resulting `Model type {x} not supported.`
  crash classifies as `UNSUPPORTED_ARCH`.

Live sites: `_installed_mlx_vlm_model_types`, `_mlx_vlm_model_remapping`,
`_model_arch_precheck`, `_arch_precheck_for_model` in `src/check_models.py`;
locked by `src/tests/test_model_discovery.py`.

## Prefer in-repo tools (do not fork cache logic)

```bash
conda activate mlx-vlm

# What default discovery would run (no generation)
cd src && python -m check_models --dry-run

# Or call the same filter helpers used in production/tests
python - <<'PY'
from check_models import get_cached_model_ids
for model_id in get_cached_model_ids():
    print(model_id)
print(f"\n{len(get_cached_model_ids())} supported model(s)")
PY
```

Implementation live site: `get_cached_model_ids` / eligibility helpers in
`src/check_models.py` under model processing / cache scan. Tests that lock the
filter: `src/tests/test_model_discovery.py`.

When `--models` is omitted, unsupported cached repos should be reported with
skip reasons (for example missing `tokenizer_config.json` or safetensors).

## Optional server cross-check

Only when validating server visibility (not needed for ordinary benchmarks):

```bash
python -m mlx_vlm.server --port 8080
curl -s http://127.0.0.1:8080/v1/models
```

Compare IDs to `get_cached_model_ids()` / `--dry-run`. Do not start the server
for routine `check_models` runs; the harness uses direct generation for
isolation.

## Reporting checklist

When reporting cache contents, include:

- cache directory if non-default (`HF_HOME` / `HUGGINGFACE_HUB_CACHE`)
- count of supported models
- exact model IDs
- whether the list came from `check_models` discovery or from
  `curl …/v1/models`
- skip reasons for excluded cached repos when relevant

## Rules

- **Do not** reimplement `scan_cache_dir` filters in ad-hoc scripts when
  `get_cached_model_ids` already encodes the contract.
- **Do not** use `uv run`.
- Explicit `--models` bypasses the filter and may include unsupported layouts;
  say so when diagnosing “works with --models but not default scan”.
- Cache presence ≠ VLM success; use `native-mlx-vlm-repro` after discovery.
