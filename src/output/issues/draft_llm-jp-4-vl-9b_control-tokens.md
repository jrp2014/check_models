# llm-jp-4-vl-9b: declared-special Harmony tokens `<|channel|>`/`<|message|>` leak into generated text (no channel parsing for `llmjpvl`)

(The H1 above is the suggested issue title; the body starts at Summary.)

## Summary

With `mlx-community/llm-jp-4-vl-9b-mlx-4bit`, tokens declared `special: true`
in the checkpoint's tokenizer.json (`<|channel|>`, `<|message|>`) are decoded
into the visible output, and the `llmjpvl` processor exposes no
`clean_output`/Harmony channel parser, so the raw channel wrapper reaches the
user instead of a parsed response.

## Environment

- MLX-VLM: 0.7.11 @ `918763c6875c80e82f9f48344dfad4763346fea8` (editable install from git)
- mlx: 0.32.5.dev20261030+1ba7f969a (editable local build)
- transformers: 5.22.1
- Python: 3.14.7
- OS / macOS: macOS 27.2 (Darwin 27.2.0)
- Hardware / MLX device: Apple M5 Max (Metal), 128 GB RAM
- Install method: editable path (git checkout)
- check_models (optional context): 0.17.98 @ `b93edf99`

## Model

- Model: `mlx-community/llm-jp-4-vl-9b-mlx-4bit`
- Resolved revision: `9c056d48b1e611dc586139a5deb927ae363cfe6f`
- Source: HF cache
- Trust remote code: yes (processor/model classes load from the checkpoint)
- Family README: none (`mlx_vlm/models/llmjpvl/` has no README; model_type `llmjpvl` is supported)

## Reproduction

The image below is a 719x1,024 downscaled re-encoding of the original input;
the failure reproduces identically on it (the original is a 63.1 MP local
photo, not published — output on both is byte-identical).

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-6973c355665180f8.jpg
printf '%s\n' '6973c355665180f82f14afd2e54719e3aaaac02b29f208456ee34bee47108151  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --verbose \
  --model mlx-community/llm-jp-4-vl-9b-mlx-4bit \
  --revision 9c056d48b1e611dc586139a5deb927ae363cfe6f \
  --image repro-image.jpg \
  --prompt 'Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-17 15:44:34 UTC+01:00
- GPS: 51.751667°N, 1.252222°W

Descriptive hints:
- Description hint: A view of a historic stone gothic tower and rooftops partially framed by dense green trees under an overcast sky.
- Keyword hints: Architecture, Branch, Building, Chimney, Cloud, Cloudy sky, England, English countryside, Foliage, Historic Building, Leaves, Outdoors, Overcast, Oxford, Oxfordshire, Pinnacle, Roof, Rooftops, Scenic, Sky

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:' \
  --max-tokens 1000 --temperature 0.0 --top-p 1.0 --top-k 0 --seed 0 \
  --prefill-step-size 2048 --trust-remote-code
```

## Expected behavior

Special tokens declared in the checkpoint's tokenizer.json are not decoded
into the user-visible text: either the Harmony channel wrapper is parsed (the
chat template declares `# Valid channels: analysis, commentary, final`), or at
minimum the `special: true` tokens are stripped from the decoded output.
mlx-vlm already handles this wrapper for other families: issue #2121 reported
the same leak for `muse_glimmer` and PR #2129 fixed it with a family
`clean_output`; PR #2329 parses Harmony channels into reasoning and content
for gpt-oss; and `mlx_vlm/tools/parsers/harmony.py` parses
`<|channel|>commentary …<|call|>` tool calls — but the `generate`/CLI output
path leaves the wrapper visible for `llmjpvl`, whose processor has no
`clean_output`.

## Actual behavior

The complete generated text, as printed by the CLI (and as returned through
`stream_generate`), is:

```text
<|channel|> analysis<|message|> The image shows a view of a historic stone gothic tower and rooftops partially framed by dense green trees under an overcast sky.
```

- `<|channel|>` (id 9) and `<|message|>` (id 12) are `special: true` in the
  checkpoint's tokenizer.json, yet appear verbatim in the decoded text.
- `getattr(processor, "clean_output", None)` is `None` for this family, so
  there is no post-processing hook that could strip or parse the wrapper.
- Generation is 29 tokens; prompt is 2,186 tokens (verbose stats below).

## Token-level stop evidence (what is *not* an mlx-vlm bug)

A `load` → `apply_chat_template` → `stream_generate` probe with the same
settings shows the last five generated token ids are
`[672, 78175, 2921, 611, 2]`, of which the last three decode to `sky`, `.`
and `<|return|>`, with `finish_reason: "stop"` and stopping criteria
`eos_token_ids == [2]`:

- The model itself emits `<|return|>` (id 2, the declared EOS) immediately
  after the analysis-channel message. `<|end|>` (id 11) is never generated and
  is not registered as a stop token, so mlx-vlm did not cut generation short.
- The checkpoint therefore never opens a `final` channel for this prompt; the
  missing requested sections are observed checkpoint behaviour, reported here
  only as context for the wrapper leak, not as a runtime defect claim.

Checkpoint facts, for reference: `generation_config.json` declares
`eos_token_id: [2, 2]`; tokenizer_config.json's `eos_token` is `<|return|>`;
all five Harmony tokens (`<|start|>` 10, `<|end|>` 11, `<|channel|>` 9,
`<|message|>` 12, `<|return|>` 2) are declared `special: true` in
tokenizer.json; the chat template (chat_template.jinja) itself renders
`<|channel|>` and `<|end|>` and instructs that a channel must be included in
every message.

## Logs / traceback

No exception; CLI verbose tail:

```text
<|channel|> analysis<|message|> The image shows a view of a historic stone gothic tower and rooftops partially framed by dense green trees under an overcast sky.
==========
Prompt: 2186 tokens, 2821.575 tokens-per-sec
Generation: 29 tokens, 102.766 tokens-per-sec
Peak memory: 6.714 GB
```

## Inputs

- Confirmed natively on 2026-10-30 against mlx-vlm git HEAD `918763c68` with
  the exact command above.
- Shareable: the downscaled preview above (JPEG, 719x1,024, 179,647 bytes,
  SHA-256 `6973c355665180f8…`), which reproduces the behaviour exactly.
- Original (not published): JPEG, 6,656x9,483 (63.1 MP), 60,507,861 bytes,
  SHA-256 `ecdd79a7ae793ce2…`; output on the original is byte-identical to the
  preview's.
- Context: the behaviour has recurred across 33+ consecutive harness sweeps
  since 2026-09-12 or earlier (same checkpoint revision;
  mlx/mlx-vlm/transformers versions varying).
