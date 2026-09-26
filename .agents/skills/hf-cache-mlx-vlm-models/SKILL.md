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

## Runnable, misconfigured, or unusable

Sweeps have shown three distinct outcomes; keep them apart when judging a
candidate or explaining a result.

**Misconfigured checkpoint** (fails, or misbehaves, whatever the model's
quality). Signals seen so far, each with its check:

| Signal | What happened | Check |
| --- | --- | --- |
| No `preprocessor_config.json` / `processor_config.json` | the processor fell back to defaults silently (a video patch size of 2 doubled the patches and crashed prefill) | "Snapshot notes" in diagnostics |
| Weight keys flattened or renamed | load rejected hundreds of weights | a native `load()` |
| Text-only chat template on a vision model | list-content messages fail at prefill, or the image is dropped | `tools.hub_precheck` template shape |
| `config.json` unparseable or without `model_type` | nothing can load it | discovery layout skip |
| `model_type` without an installed loader package | load crash (`Model type … not supported`) | architecture pre-check |
| Weights larger than unified memory | cannot be held | `tools.hub_precheck` memory verdict |
| Missing or empty weight shards | incomplete download | discovery layout skip |

The remedy is usually another checkpoint, not a code fix. The family README
in the installed mlx-vlm (`mlx_vlm/models/<family>/README.md`, named in issue
drafts) can list a corrected repo; the `nativ-community` checkpoints linked
from them loaded and ran where the `mlx-community` copies did not.

**Runnable but unusable output** is a finding in its own right, not a reason
to drop a model: missing requested fields, repetition loops, unfinished
thinking blocks, or leaked control tokens. Keep such models in the sweep
(they are often the only coverage for their architecture) and report them.
Leaked control or role tokens may point at a template or tokenizer config
rather than the model; classify them only after a native reproduction
(`native-mlx-vlm-repro`).

**Environmental, not the model:** the first sweep after an mlx rebuild runs
on a cold Metal shader cache and inflates prefill (up to 4x on one model;
under a second for most); concurrent CPU work, including a git history
search, dents throughput for the models running at the time. Compare timing
on a warm, idle machine.

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
