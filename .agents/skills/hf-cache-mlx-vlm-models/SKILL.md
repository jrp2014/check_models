---
name: hf-cache-mlx-vlm-models
description: >
  Explains which cached Hugging Face models the harness selects, why others
  are skipped, and how hub checkpoints are checked before download. Use for
  cached-model discovery, skipped repositories, dry-run selections, cache
  paths, pre-download checks, or differences from mlx-vlm server listings.
  Cache eligibility is not proof that a model can generate.
---

# Hugging Face Cache Discovery

Explain which cached models this harness would select, why others are skipped,
and what the available evidence can establish. Use the repository's discovery
logic rather than inventing a parallel cache scanner.

Adapted from upstream
[`hf-cache-models`](https://github.com/Blaizzy/mlx-vlm/tree/main/skills/skills/hf-cache-models).
Follow the [canonical project guidance](../../../.github/copilot-instructions.md)
for the conda environment and validation workflow; use conda + pip, not uv.

## Establish the question and evidence

For a listing or diagnosis, inspect the effective cache directory, installed
package versions, and current discovery results. Start with retained diagnostics
when they answer the question. A request to explain a skip does not authorize
downloads, cache cleanup, inference, or starting a server.

Distinguish these independent decisions:

| Evidence | Meaning |
| --- | --- |
| Cache layout | Required files exist on the cached main revision |
| Image capability | Metadata identifies image-input support, another purpose, or uncertainty |
| Architecture pre-check | The installed mlx-vlm appears to recognise the model family |
| Native generation | An actual bounded run succeeds or fails under recorded conditions |

Default selection requires an eligible layout and no confident negative
image-capability finding. Unknown capability remains selected with a warning.
An unsupported architecture is advisory, not proof of failure or a reason to
silently remove a model from the sweep.

Explicit `--models` bypasses default discovery selection, not runtime validity
checks. It can therefore attempt a model that default discovery would skip.

## Cache layout and shared files

The layout check requires a model repository with a cached `main` revision,
`config.json`, `tokenizer_config.json`, and either a safetensors index or
safetensors weights. Indexed weights must also be complete and readable.
This is a file-presence check, not a generation test.

Snapshots normally contain links. Depending on the installed Hub version,
targets may live in the repo's blob store or the hub-wide
`blobs/<xx>/<hash>` store shared by repositories. A target outside the
individual repo is not by itself missing or corrupt.

When investigating missing files, trace the link to a regular readable file and
check the allowed cache roots. Keep shared-store support distinct from trusting
arbitrary neighbouring directories. Plain local model directories have their
own containment boundary. Preserve those distinctions in regression tests.

## Capability and architecture

Use bounded metadata reads and evidence from the installed implementation.
Repo names, familiar model families, and the presence of weights are not enough
to establish image-input support.

- Positive vision metadata can establish image capability.
- Positive evidence of another purpose can justify exclusion.
- Insufficient or contradictory evidence remains unknown.
- Architecture support is checked against the installed package, whose coverage
  and aliases can differ from another release.

Find current rules through the cache-discovery section of
`src/check_models.py` and `src/tests/test_model_discovery.py`. Prefer those
sources to maintaining a second list of classifier fields, model families, or
exceptions here.

## Choose the smallest relevant check

From the repository root:

```bash
conda activate mlx-vlm
(cd src && python -m check_models --dry-run)
```

Dry-run exercises normal CLI setup and needs a valid image input. It does not
invoke models, but it still imports runtime dependencies; an import failure is
not a model-discovery verdict. For cache-only inspection, use the existing
discovery helpers rather than requiring an inference run.

When the user asks about a candidate before downloading, use the existing Hub
pre-check tool. It reads remote file lists and small metadata files, not weights:

```bash
(cd src && python -m tools.hub_precheck org/model)
```

Check the installed tool's help for current options. Its layout, architecture,
and template checks are hints, not proof of generation. For server-listing
comparisons, establish whether the server uses served-model or cache discovery;
start a server only when the requested task needs it. For actual reproduction,
use `native-mlx-vlm-repro`.

## Report the result

Give the effective cache location, exact selected IDs, counts, and skip reasons
relevant to the request. Name the source of the listing and separate selection
from architecture support and demonstrated generation. If the reason is not
established, say what evidence is missing rather than guessing from a model name.

For discovery changes, test both accepted cache layouts and rejected escaping or
missing targets using temporary fixtures. Do not alter the user's cache to
prove a fix.

## Cache removal is a separate action

Only remove entries when the user asks. Use the Hub CLI's supported cache
commands, checking their current help and previewing the intended targets:

```bash
hf cache rm model/org/name --dry-run
```

Do not recursively delete `models--…` directories: shared blobs may remain
or be referenced by another repository. Pruning is broader cleanup, not an
automatic sequel to removing one model; inspect its scope and obtain
authorization for anything beyond the requested removal.
