# Draft upstream issue: `--thinking-budget` is silently disarmed when the model generates `<think>` instead of the template seeding it

Improved companion to the generalised thinking-controls draft
(`UPSTREAM_THINKING_CONTROLS_ISSUE_DRAFT.md`). Paste-ready below the line.

---

## Summary

`ThinkingBudgetCriteria` already contains the logic to start counting when the
model *generates* the configured opening delimiter — but
`generate/dispatch.py` disarms that logic before it can run, by overwriting
the caller's `enable_thinking` with a "was the opener already in the prompt?"
check. The result: for exactly the models whose templates do not seed
`<think>` (the GLM Thinking family, for example), an accepted
`--thinking-budget` has no effect, and the CLI exits successfully with no
indication that the budget was inactive.

## What the problem is

`enable_thinking` is doing double duty inside `ThinkingBudgetCriteria`:

1. **Caller intent** — "enforce a thinking budget", and
2. **Initial decoding state** — "the prompt already ends inside an open
   thinking block" (`self.in_thinking = self.enable_thinking`,
   `mlx_vlm/utils.py:2095` and `:2102`).

`generate/dispatch.py:1022-1028` resolves that ambiguity by collapsing the
caller's flag into the prompt check:

```python
if thinking_budget is not None:
    thinking_start_token_id = tokenizer.encode(
        thinking_start_token, add_special_tokens=False
    )[-1]
    enable_thinking = enable_thinking and (
        thinking_start_token_id in input_ids.flatten().tolist()
    )
```

When the template does not seed the opener, `enable_thinking` reaches the
criteria as `False` — which disarms the *already-implemented* generated-opener
transition in `ThinkingBudgetCriteria.__call__`
(`mlx_vlm/utils.py:2107`):

```python
if self.enable_thinking and token_id == self.thinking_start_token_id:
    self.in_thinking = True
    return None
```

So the state machine that would handle a generated `<think>` correctly can
never fire for the models that need it most: checkpoints that emit their
opener during generation think unbounded, while the budget option is accepted
without complaint.

## Minimal fix

Pass the two facts separately instead of conflating them:

- `enable_thinking` = the caller's request, unmodified;
- initial `in_thinking` = whether `thinking_start_token_id` occurs in
  `input_ids` (the current prompt check, used only for state
  initialisation).

With that split, `__call__` works unchanged for both seeded and generated
openers: a seeded prompt starts inside the block and counts immediately; a
generated opener enters the block via the existing transition at
`utils.py:2107` and then counts. If neither ever occurs, a warning that the
budget was not activated would let callers distinguish "enforced" from
"inactive".

## Environment

- **mlx-vlm:** 0.6.11, editable checkout at `0c846ef211599d4a2036e7b7435e6ff16f0d6bce`
- **mlx:** 0.32.1.dev20260808+8d6662986, editable checkout at `8d666298652fdac2e7727ecdcf507b1d199bba16`
- **mlx-lm:** 0.31.3, editable checkout at `254d153fdeb6f150edd4fc5a54f9828638481fa8`
- **transformers:** 5.14.1
- **Python:** 3.13.13; **macOS:** 26.6; **Hardware:** Apple M5 Max, Metal, 128 GB
- **Install method:** conda environment with pip/editable installs

## Model

- **Model:** `mlx-community/GLM-4.1V-9B-Thinking-8bit`
- **Resolved revision:** `9677807f106500eb7690391c27645d59f6855cfb`
- **Source:** Hugging Face cache; **Trust remote code:** enabled

## Reproduction

The image is the public `examples/images/cats.jpg` file from the mlx-vlm
checkout: 640 x 480 JPEG, SHA-256
`dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`.

```bash
python -m mlx_vlm.generate \
  --model mlx-community/GLM-4.1V-9B-Thinking-8bit \
  --revision 9677807f106500eb7690391c27645d59f6855cfb \
  --image examples/images/cats.jpg \
  --prompt 'Describe the image, then return exactly: Title, Description, and Keywords.' \
  --max-tokens 500 \
  --temperature 0.0 \
  --prefill-step-size 2048 \
  --enable-thinking \
  --thinking-budget 50 \
  --thinking-start-token '<think>' \
  --thinking-end-token '</think>' \
  --trust-remote-code \
  --verbose
```

The rendered prompt ends as follows and does not seed `<think>` (confirmed
independently via `apply_chat_template` with default kwargs — this template
never seeds the opener):

```text
Describe the image, then return exactly: Title, Description, and Keywords.<|assistant|>
```

The generated response starts with `<think>`, continues far beyond 50
thinking tokens, closes the block naturally, and finishes after 480 generated
tokens:

```text
<think>Got it, let's start by analyzing the image. There are two cats lying
on a pink blanket, which is on a red couch. ...

Now, make sure the title is concise. Description is detailed but not too long.
Keywords are relevant terms.</think><answer>Title: "Cats Relaxing on a Pink
Blanket with Remote Controls"
...
```

```text
Prompt: 414 tokens
Generation: 480 tokens
```

Removing all four thinking options produces the identical rendered prompt,
generated text, and 480-token generation at temperature 0.0 — i.e. the four
options changed nothing observable.

## Expected behaviour

With `--enable-thinking --thinking-budget 50` accepted, either:

1. counting starts when the model emits the configured opening delimiter, and
   the configured closing delimiter is forced once the budget is exceeded
   (the behaviour `ThinkingBudgetCriteria` already implements); or
2. the CLI reports that the budget was not activated for this prompt/model.

Silently generating an order of magnitude more thinking tokens than requested
makes it impossible for callers to reserve part of `max_tokens` for the final
answer.

## Actual behaviour

The budget has no observable effect when `<think>` is absent from the input
but emitted during generation. The CLI exits successfully, so callers cannot
distinguish an enforced budget from an inactive one.

## Broader integration evidence

An integration survey over thinking-capable checkpoints (details in the
companion report) shows this is the common case, not an edge case. First with
default options and a 500-token allowance, six variants (Qwen3-VL Thinking,
MiniCPM-V, Kimi-VL, ERNIE, GLM-4.1V Thinking, Apriel Thinker) spent the
entire allowance on visible reasoning without returning the requested output.
A follow-up with `enable_thinking=true`, `thinking_budget=300`, and
`max_tokens=1000`:

| Model family | Opener provenance | Follow-up outcome |
| --- | --- | --- |
| Qwen3-VL Thinking | template-seeded | Closed `<think>`, complete answer in 428 tokens |
| MiniCPM-V | template-seeded | Closed `<think>`, complete answer in 412 tokens |
| ERNIE | template-seeded | Closed `<think>`, answered, then repeated to the 1,000 cap |
| GLM-4.1V Thinking | **generated** | Unclosed `<think>` to the 1,000 cap — budget inactive |
| GLM-4.6V (nvfp4) | **generated** | Unclosed to the 1,000 cap (its mxfp4 sibling closed at 420) |
| Kimi-VL | generated `◁think▷` | Closed naturally at 463 tokens (default pair cannot match) |
| Apriel Thinker | none (unmarked prose) | 1,000 cap (no delimiter for any control to act on) |

The seeded group — the one case the current gate arms — behaved; the
generated-opener group ran unbounded. Kimi and Apriel illustrate the related
delimiter-mismatch and unmarked-reasoning limitations covered in the
companion report.

## Related default-semantics question

The same native GLM command without `--enable-thinking` still emits the
complete `<think>...</think><answer>...` response. That may be intended for
an explicitly named Thinking checkpoint; if so, documenting that
`enable_thinking=false` cannot disable reasoning for every checkpoint would
help callers distinguish template control from model behaviour.
