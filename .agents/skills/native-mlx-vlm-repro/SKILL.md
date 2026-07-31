---
name: native-mlx-vlm-repro
description: >
  Run or debug native mlx-vlm CLI/Python inference outside check_models when
  verifying harness findings, minimizing reproductions, or confirming upstream
  regressions. Use for python -m mlx_vlm.generate, load/apply_chat_template/
  generate scripts, processor or chat-template errors, media-input failures, and
  deterministic native repro commands. Prefer existing issue-draft or diagnostics
  repro blocks when present. This repo uses conda + pip, never uv.
---

# Native mlx-vlm Reproduction

Use this workflow to isolate failures to **upstream mlx-vlm** rather than the
`check_models` harness. Prefer native commands over extending the harness when
the goal is an upstream-ready repro.

Adapted from upstream mlx-vlm support skills
([Blaizzy/mlx-vlm#1343](https://github.com/Blaizzy/mlx-vlm/pull/1343)), with this
repo’s package manager and artifact conventions.

## Environment (required)

```bash
conda activate mlx-vlm
cd src && python -m tools.validate_env && cd ..
python -m mlx_vlm.generate --help   # confirm installed CLI flags before finalizing a repro
```

- Use the **conda `mlx-vlm` env** and **`pip`** installs only.
- Prefer `python -m mlx_vlm.generate` / `python -m mlx_vlm.server` over bare
  entry-point scripts when documenting commands.
- **Never** document or run `uv run …` in this repository.
- Do not download large models unless the user asks; prefer local cache paths or
  already-cached HF IDs.

## First checks

1. Identify model ID or local path, modality, prompt, media files, and expected
   output.
2. Prefer an existing local/cached model when reproducing.
3. If `check_models` already produced artifacts, start from those instead of
   inventing a new command:
   - crash draft: `src/output/issues/issue_*.md` (CLI + Python blocks)
   - aggregate: `src/output/reports/diagnostics.md` → Shared Reproduction
   - machine facts: `src/output/results.jsonl`, `src/output/run.json`
4. Confirm flags with `python -m mlx_vlm.generate --help` before treating a
   command as final; mlx-vlm flags drift across releases.
5. When debugging a model family inside an mlx-vlm checkout, read
   `mlx_vlm/models/<family>/README.md` if present.

## Command patterns

Text-only smoke:

```bash
python -m mlx_vlm.generate \
  --model <model-or-path> \
  --prompt "Write a short answer." \
  --max-tokens 128 \
  --temperature 0.0
```

Image (usual VLM path for this project):

```bash
python -m mlx_vlm.generate \
  --model <model-or-path> \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 128 \
  --temperature 0.0
```

Pin revision and trust-remote-code when the harness did:

```bash
python -m mlx_vlm.generate \
  --model <model-or-path> \
  --revision <resolved-sha-or-ref> \
  --image /path/to/image.jpg \
  --prompt-file /path/to/prompt.txt \
  --max-tokens 500 \
  --temperature 0.0 \
  --trust-remote-code
```

Canonical Python shape used in issue drafts (one model per process):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

model, processor = load(MODEL, **LOAD_KWARGS)
formatted = apply_chat_template(
    processor, model.config, PROMPT, num_images=1, **TEMPLATE_KWARGS
)
if isinstance(formatted, list):
    formatted = "\n".join(str(message) for message in formatted)
result = generate(model, processor, formatted, image=IMAGE, **GENERATE_KWARGS)
print(result.text)
```

## Reproducibility rules

- Include exact command, model ID/path, resolved revision, media type/size,
  Python version, package versions or git commits, and full error/traceback.
- Use greedy / low temperature (`--temperature 0.0`) when debugging quality or
  regressions.
- Bound output with `--max-tokens`.
- Preserve shell quoting exactly (JSON prompts, thinking delimiters, newlines).
- Prefer portable paths in filed issues; strip private home directories when
  pasting externally.
- If failure depends on media, record dimensions (and duration/codec for
  audio/video) and whether a small synthetic input still fails.
- Run **one model per process** to avoid sequential Metal-state interactions.
- Prefer the harness’s effective generation kwargs from diagnostics/JSONL over
  guessed flags. Only include native-CLI-supported settings in CLI repros;
  put harness-only settings in the Python block or prose.

## Failure routing

| Symptom | Where to look |
| ------- | ------------- |
| `model_type` unsupported | `config.json`; mlx-vlm `mlx_vlm/models/` |
| Missing weights | `.safetensors` or `model.safetensors.index.json` in cache snapshot |
| Processor / chat template errors | processor configs, `prompt_utils`, model-family README |
| Media shape errors | single image first, then multi-input |
| Works in native CLI, fails in harness | harness preflight, prompt construction, kwarg mapping |
| Fails natively the same way | upstream issue path (`upstream-mlx-vlm-issues` skill) |

## Validation

- Smallest command that exercises the failing modality.
- Do **not** create ad-hoc test scripts in this repo; either use native mlx-vlm
  commands, the parameterised repro from diagnostics, or add a focused pytest
  under `src/tests/`.
- For local mlx-vlm checkouts only: focused upstream tests such as
  `test_generate.py` / `test_cli.py` — still with `pip`/`pytest`, not `uv`.
- After confirming the native repro, switch to
  `upstream-mlx-vlm-issues` if the user wants issue-ready Markdown.
