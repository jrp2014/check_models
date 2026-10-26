---
name: native-mlx-vlm-repro
description: >
  Reproduces harness findings with native mlx-vlm outside check_models:
  `python -m mlx_vlm.generate` commands and load, apply_chat_template and
  generate scripts. Use when verifying a harness finding, minimising a
  reproduction, confirming an upstream regression, or debugging processor,
  chat-template or media-input errors.
---

# Native mlx-vlm Reproduction

Use this workflow to isolate failures to **upstream mlx-vlm** rather than the
`check_models` harness. Prefer native commands over extending the harness when
the goal is an upstream-ready repro.

Adapted from the upstream `cli-inference` and `reproducible-github-issues`
skills ([`skills/skills/`](https://github.com/Blaizzy/mlx-vlm/tree/main/skills/skills), added by
[Blaizzy/mlx-vlm#1747](https://github.com/Blaizzy/mlx-vlm/pull/1747)), with this
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
- Write and run commands as `python -m …`, not `uv run …`: the environment is conda-managed, and uv would install outside it.
- Do not download large models unless the user asks; prefer local cache paths or
  already-cached HF IDs.

## First checks

1. Identify model ID or local path, modality, prompt, media files, and expected
   output.
2. Read the model family's `mlx_vlm/models/<family>/README.md` (in the
   editable checkout or the installed package) before anything else. It may
   name a corrected or recommended checkpoint (for example the
   `nativ-community/` repos listed for Mage-VL, Mistral 3 and Nemotron 3 Nano
   Omni). Reproduce the original checkpoint and resolved revision first;
   testing a recommendation is a subsequent controlled comparison, within
   the user's download and inference authorization, not a substitute.
3. Prefer an existing local/cached model when reproducing.
4. If `check_models` already produced artifacts, start from those instead of
   inventing a new command:
   - sweep overview: `src/output/issues/run_summary.md`
   - crash draft: `src/output/issues/issue_*.md` (native CLI command + traceback)
   - aggregate: `src/output/reports/diagnostics.md` → Shared Reproduction
   - machine facts: `src/output/results.jsonl` (the metadata header carries the
     run context: image digest, generation settings, provenance, comparison)
5. Confirm flags with `python -m mlx_vlm.generate --help` before treating a
   command as final; mlx-vlm flags drift across releases.

## Command patterns

These are smoke-test shapes, not replacements for a retained run's settings.
For a reported failure, first copy its effective prompt, decoding settings,
token limit, checkpoint revision and media. Change one factor at a time only
after that baseline is established.

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

For a streaming-specific investigation, this Python probe skips draft text,
keeps final-chunk metrics and applies optional output cleanup, as the harness
does. It does not reproduce the harness's repetition abort or instrumentation.
Use one model per process and fill the uppercase inputs from retained evidence:

```python
from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

model, processor = load(MODEL, **LOAD_KWARGS)
formatted = apply_chat_template(
    processor, model.config, PROMPT, num_images=1, **TEMPLATE_KWARGS
)
if isinstance(formatted, list):
    formatted = "\n".join(str(message) for message in formatted)
# generate() and the harness register --eos-tokens before streaming;
# stream_generate alone does not, so a repro of such a run must do it too.
criteria = getattr(getattr(processor, "tokenizer", processor), "stopping_criteria", None)
if criteria is not None and EOS_TOKENS:
    criteria.add_eos_token_ids(EOS_TOKENS)
pieces, last = [], None
for chunk in stream_generate(
    model=model, processor=processor, prompt=formatted, image=IMAGE, **GENERATE_KWARGS
):
    last = chunk
    if getattr(chunk, "is_draft", False):
        continue  # speculative/diffusion drafts are display only
    pieces.append(chunk.text)
text = "".join(pieces)
if last is not None:
    clean_output = getattr(processor, "clean_output", None)
    if callable(clean_output) and isinstance(cleaned := clean_output(text), str):
        text = cleaned
print(text)
if last is not None:
    print(last.generation_tps, last.peak_memory, last.finish_reason)
else:
    print("No chunks returned; no final-chunk metrics.")
```

Prefer the generated native CLI command or upstream `generate()` for an
end-to-end reproduction; use the probe above when the streaming boundary is
what needs isolation. The harness calls `stream_generate` directly but mirrors
wrapper behaviour including `processor.clean_output`. Distinguish raw streamed
text from the final cleaned answer when reporting an output discrepancy.

## Reproducibility rules

- Include exact command, model ID/path, resolved revision, media type/size,
  Python version, package versions or git commits, and full error/traceback.
- Preserve the failing run's effective decoding settings first. Greedy decoding
  (`--temperature 0.0`) can be a separate control experiment; it can hide a
  sampling-specific regression and must not silently replace the reproduction.
- Bound output with `--max-tokens`.
- Preserve shell quoting exactly (JSON prompts, thinking delimiters, newlines).
- Prefer portable paths in filed issues; strip private home directories when
  pasting externally.
- Treat a retained gallery `source-image` as a sanitised preview unless its
  digest matches the recorded inference input. A public URL plus matching
  SHA-256 can support an exact download-and-verify command. For local-only media,
  report format, dimensions, byte size, and digest without claiming a complete
  reproduction command.
- If failure depends on media, record dimensions (and duration/codec for
  audio/video) and whether a small synthetic input still fails.
- Run **one model per process** to avoid sequential Metal-state interactions.
- Prefer the harness’s effective generation kwargs from diagnostics/JSONL over
  guessed flags. Only include native-CLI-supported settings in CLI repros;
  put harness-only settings in the Python block or prose.
- State sampling explicitly. Before mlx-vlm 73d366e the native CLI ignores
  the checkpoint's `generation_config.json`
  ([Blaizzy/mlx-vlm#2228](https://github.com/Blaizzy/mlx-vlm/issues/2228));
  from 73d366e it fills omitted `--temperature`, `--top-p` and `--top-k` from
  it. Either way an omitted flag may not be what the harness ran, so copy all
  three from the model's effective `generate_kwargs` (the harness's own repro
  commands already do).

## Thinking-output checks

- A properly closed configured thinking block followed by substantive final text
  is valid output evidence, not a failure by itself.
- Check the rendered prompt for a seeded opening delimiter before treating a
  generated closing delimiter as stray output.
- Reproduce natively when the block is unclosed, consumes the token budget, lacks
  a final answer, or exposes an undeclared control/role token. Name the specific
  condition that is unexpected; “thinking present” is not a defect on its own.

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
- Record the outcome against the retained run with
  `cd src && python -m tools.add_native_repro_note MODEL OBSERVATION
  confirmed|not_reproduced --checker NAME --evidence "command and result"`.
  The run summary's *Native repro* column then reads confirmed or not
  reproduced until the model revision or the mlx/mlx-vlm/transformers build
  changes. Record only a check that was actually run natively.
