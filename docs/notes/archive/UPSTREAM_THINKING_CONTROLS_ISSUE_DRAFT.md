# Draft upstream issue: thinking controls do not reliably enable, disable, bound, or detect reasoning across thinking-capable models

Paste-ready generalisation of the single-model `--thinking-budget` report
(companion draft: "`--thinking-budget` is ignored when the model generates
`<think>` after prefill"). Evidence below comes from two aggregate
check_models runs over the same 42-model cache plus native template probes;
the companion draft holds the isolated single-command reproduction.

---

## Summary

Across ten thinking-capable checkpoints, visible reasoning is emitted largely
independently of the caller's `enable_thinking` / `thinking_budget` /
`thinking-*-token` settings: some chat templates seed an open `<think>` when
thinking was never requested, some models re-open or continue reasoning after
their template emits the disabled convention, some generate their own opening
delimiter (which silently deactivates `thinking_budget` via the `input_ids`
gate in `generate/dispatch.py`), some use delimiters the defaults cannot
match, and some reason with no delimiters at all. The practical result is
that callers cannot reserve part of `max_tokens` for a final answer, cannot
turn reasoning off, and get no signal that any of the controls were inactive.

## Environment

- **mlx-vlm:** 0.6.11, editable checkout at `0c846ef211599d4a2036e7b7435e6ff16f0d6bce`
- **mlx:** 0.32.1.dev20260808+8d6662986, editable checkout at `8d666298652fdac2e7727ecdcf507b1d199bba16`
- **mlx-lm:** 0.31.3, editable checkout at `254d153fdeb6f150edd4fc5a54f9828638481fa8`
- **transformers:** 5.14.1
- **Python:** 3.13.13; **macOS:** 26.6; **Hardware:** Apple M5 Max, Metal, 128 GB
- **Install method:** conda environment with pip/editable installs

## What the chat templates actually render (no thinking options passed)

Rendered-prompt tails from `apply_chat_template(processor, config, prompt,
num_images=1)` with **no** thinking kwargs (so `enable_thinking` takes
upstream's default `False` where the template supports it). Reproducible
without loading model weights:

```python
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template
from transformers import AutoProcessor

for model in (
    "mlx-community/Qwen3-VL-2B-Thinking-bf16",
    "mlx-community/MiniCPM-V-4.6-8bit",
    "mlx-community/GLM-4.1V-9B-Thinking-8bit",
):
    proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    cfg = load_config(model, trust_remote_code=True)
    print(repr(apply_chat_template(proc, cfg, "Describe the image.", num_images=1)[-50:]))
```

| Model | Rendered tail (thinking disabled) | Template behaviour |
| --- | --- | --- |
| `Qwen3-VL-2B-Thinking-bf16` | `…<\|im_start\|>assistant\n<think>\n` | Seeds an **open** `<think>` — reasoning on regardless of flag |
| `ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | `…Assistant: \n<think>\n` | Seeds an **open** `<think>` |
| `MiniCPM-V-4.6-8bit` | `…assistant\n<think>\n\n</think>\n\n` | Emits the **disabled convention** (empty closed pair) |
| `Step-3.7-Flash-oQ2e` | `…assistant\n<think>\n</think>\n\n` | Disabled convention |
| `GLM-4.6V-Flash` (mxfp4/nvfp4) | `…Keywords:/nothink<\|assistant\|>\n<think></think>\n` | Disabled convention plus explicit `/nothink` |
| `GLM-4.1V-9B-Thinking-8bit` | `…Describe the image.<\|assistant\|>\n` | Seeds nothing — the model generates its own opener |

## Observed generation outcomes

Two aggregate runs over the same cached models, same 42-model matrix, one
high-resolution photograph and a metadata-cataloguing prompt per run (images
are local and not published; the companion draft reproduces the key case with
the public `examples/images/cats.jpg`). Temperature 0.0 throughout.

**Run A — no thinking options, `max_tokens=500`:**

| Model | Opener provenance | Outcome |
| --- | --- | --- |
| `Qwen3-VL-2B-Thinking-bf16` | template-seeded (open) | Unmarked reasoning prose, hit 500 cap, no complete metadata |
| `ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | template-seeded (open) | Unmarked reasoning, hit 500 cap |
| `MiniCPM-V-4.6-8bit` | template disabled it; model **re-opened** `<think>` | Hit 500 cap |
| `Step-3.7-Flash-oQ2e` | template disabled it; model reasoned in prose | 438 tokens, completed |
| `GLM-4.1V-9B-Thinking-8bit` | **generated** `<think>` | Hit 500 cap, block unclosed |
| `Kimi-VL-A3B-Thinking-2506-bf16` | generated `◁think▷` (own delimiters) | Hit 500 cap |
| `Apriel-1.5-15b-Thinker-6bit-MLX` | none — unmarked reasoning ("Here are my reasoning steps:") | Hit 500 cap |
| `diffusiongemma-26B-A4B` (both quants) | generated `<\|channel>thought` … `<channel\|>` | Completed (~82 tokens) |
| `GLM-4.6V-Flash-mxfp4` | `/nothink` respected | 104 tokens, direct answer |
| `GLM-4.6V-nvfp4` | `/nothink` respected in this run | 92 tokens, direct answer |

Six of ten thinking-capable variants spent the entire 500-token allowance on
reasoning and returned incomplete or no requested output, exiting
successfully.

**Run B — `enable_thinking=true`, `thinking_budget=300`, `max_tokens=1000`
(defaults `<think>`/`</think>` as the delimiter pair):**

| Model | Outcome with a 300-token budget |
| --- | --- |
| `Qwen3-VL-2B-Thinking-bf16` | Completed in 428 tokens |
| `MiniCPM-V-4.6-8bit` | Completed in 412 tokens |
| `Step-3.7-Flash-oQ2e` | Completed in 438 tokens |
| `Kimi-VL-A3B-Thinking-2506-bf16` | Own `◁think▷` pair; closed naturally; 463 tokens (budget could not apply to these delimiters) |
| `ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | Closed `</think>`, returned metadata, then repeated keywords to the 1,000 cap |
| `GLM-4.1V-9B-Thinking-8bit` | Generated opener; **unclosed `<think>` to the 1,000 cap** — budget silently inactive |
| `GLM-4.6V-nvfp4` | Generated opener; unclosed to the 1,000 cap (its mxfp4 sibling closed at 420) |
| `Apriel-1.5-15b-Thinker-6bit-MLX` | Unmarked reasoning to the 1,000 cap — no delimiter for any control to act on |

## Failure modes, generalised

1. **`enable_thinking=false` is not "off".** Two templates seed an open
   `<think>` regardless of the flag; two models (MiniCPM, Step-3.7) reason
   past their template's own disabled convention; the same family diverges by
   quantisation (GLM-4.6V mxfp4 vs nvfp4).
2. **`thinking_budget` deactivates exactly when it is most needed.** The
   `input_ids` gate in `generate/dispatch.py` downgrades `enable_thinking`
   unless the opener is already in the prompt, so models that generate their
   opener (GLM-4.1V, GLM-4.6V-nvfp4) run unbounded — the case the budget
   exists for.
3. **Delimiter defaults cannot match several model families.** Kimi's
   `◁think▷`/`◁/think▷` and diffusiongemma's `<|channel>thought`/`<channel|>`
   are already known to the server's `ThinkingStreamState` marker table, but
   the generate path only honours the caller-supplied pair, with no warning
   when it never matches.
4. **Unmarked reasoning is invisible to every control.** Apriel (and the
   seeded-opener models, whose reasoning appears unmarked in the output
   because the opener lives in the prompt) emit reasoning prose that cannot
   be bounded, stripped, or detected.
5. **All of the above are silent.** Every run exits 0; callers cannot
   distinguish an enforced budget or a disabled thinking mode from an
   inactive one.

## Suggested direction (per the companion draft's `generate/dispatch.py` analysis)

1. **Decouple caller intent from prompt state.** Keep the requested
   `enable_thinking` as caller intent; initialise the budget criteria's
   `in_thinking` from whether the rendered prompt ends inside an open
   thinking block, and let the configured (or detected) start token entered
   *during generation* transition into the thinking state. This makes
   `thinking_budget` effective for generated openers — the companion draft's
   proposed implementation.
2. **Warn when a control is inactive.** If `enable_thinking=false` but the
   rendered prompt still ends inside an open thinking block, or
   `thinking_budget` is set but the configured pair never appears while a
   known alternative pair does, emit a warning so callers can react.
3. **Reuse the server's marker table in the generate path.** The open/close
   pairs in `mlx_vlm/server/responses_state.py` would let budget and
   detection work for Kimi and channel-marker models without per-call flags.
4. **Document per-checkpoint reality.** Where a template cannot disable
   reasoning (or a checkpoint reasons without delimiters), a note that
   `enable_thinking=false` is advisory for that model would save callers the
   discovery cost.

## Isolated reproduction

The companion draft reproduces mode 2 with one native command
(`GLM-4.1V-9B-Thinking-8bit`, public `examples/images/cats.jpg`,
`--enable-thinking --thinking-budget 50` producing the identical 480-token
response as the default invocation). The template probe above reproduces mode
1 without loading weights. Aggregate tables are context from an integration
survey, not a controlled benchmark.
