# `--thinking-budget` is ignored when the model generates `<think>` after prefill

## Summary

With `mlx-community/GLM-4.1V-9B-Thinking-8bit`, the native mlx-vlm CLI
accepts `--enable-thinking --thinking-budget 50`, but generates the same
480-token response as the default invocation. The model emits `<think>` as its
first generated token; the rendered prompt does not contain that token.

This is not a report that visible reasoning is inherently wrong. The observable
problem is that the documented thinking budget is silently inactive when the
opening delimiter is generated rather than seeded by the chat template.

## Environment

- **mlx-vlm:** 0.6.11, editable checkout at
  `0c846ef211599d4a2036e7b7435e6ff16f0d6bce`
- **mlx:** 0.32.1.dev20260808+8d6662986, editable checkout at
  `8d666298652fdac2e7727ecdcf507b1d199bba16`
- **mlx-lm:** 0.31.3, editable checkout at
  `254d153fdeb6f150edd4fc5a54f9828638481fa8`
- **transformers:** 5.14.1
- **Python:** 3.13.13
- **macOS:** 26.6
- **Hardware:** Apple M5 Max, Metal, 128 GB unified memory
- **Install method:** conda environment with pip/editable installs

## Model

- **Model:** `mlx-community/GLM-4.1V-9B-Thinking-8bit`
- **Resolved revision:** `9677807f106500eb7690391c27645d59f6855cfb`
- **Source:** Hugging Face cache
- **Trust remote code:** enabled

## Native reproduction

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

The rendered prompt ends as follows and does not seed `<think>`:

```text
Describe the image, then return exactly: Title, Description, and Keywords.<|assistant|>
```

The generated response starts with `<think>`, continues far beyond 50 thinking
tokens, closes the block naturally, and finishes after 480 generated tokens:

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

Removing all four thinking options produces the same rendered prompt, generated
text and 480-token generation at temperature 0.0.

## Expected behaviour

When `--enable-thinking --thinking-budget 50` is accepted, mlx-vlm should do
one of the following:

1. start counting when the model emits the configured opening delimiter and
   force the configured closing delimiter after the budget; or
2. report clearly that the budget was not activated for this prompt/model.

Silently generating substantially more than 50 thinking tokens makes it
difficult for callers to reserve part of `max_tokens` for the final answer.

## Actual behaviour

The budget has no observable effect when `<think>` is absent from the input but
is emitted during generation. The CLI exits successfully, so callers cannot
distinguish an enforced budget from an inactive one.

## Likely cause

This is an inference from the installed source. In `generate/dispatch.py`, the
requested `enable_thinking` value is changed to false unless the encoded opening
token already occurs in `input_ids`:

```python
if thinking_budget is not None:
    thinking_start_token_id = tokenizer.encode(
        thinking_start_token, add_special_tokens=False
    )[-1]
    enable_thinking = enable_thinking and (
        thinking_start_token_id in input_ids.flatten().tolist()
    )
```

`ThinkingBudgetCriteria` can observe generated tokens, but receives
`enable_thinking=False` in this case, so a subsequently generated `<think>`
cannot activate counting.

A possible implementation would retain the caller's enabled state separately
from whether the prompt starts inside a thinking block: initialise
`in_thinking` from the rendered prompt, then allow the configured generated
start token to enter the thinking state.

## Broader integration evidence

An integration run over six thinking-capable model variants first used the
default 500-token allowance. All six exposed reasoning and reached the token cap
before returning complete catalogue metadata. A follow-up used
`enable_thinking=true`, `thinking_budget=300`, and `max_tokens=1000`:

| Model family | Follow-up observation |
| --- | --- |
| Qwen3-VL Thinking | Closed `<think>` and returned final metadata in 428 tokens |
| MiniCPM-V | Closed `<think>` and returned final metadata in 412 tokens |
| Kimi-VL | Used `◁think▷` / `◁/think▷`, then returned metadata in 463 tokens |
| ERNIE | Closed `<think>`, returned metadata, then repeated keywords to the 1,000-token cap |
| GLM-4.1V Thinking | Generated an unclosed `<think>` block to the 1,000-token cap for the longer prompt |
| Apriel Thinker | Emitted unmarked reasoning and reached the 1,000-token cap |

The two aggregate runs used different images and prompts, so this table is
context rather than a controlled comparison. The native GLM command above is
the isolated reproduction.

The wider results also suggest that callers need either model-aware thinking
delimiter defaults or an explicit signal that the configured delimiters do not
match the model. For example, Kimi's `◁think▷` delimiters cannot be controlled
by the default `<think>` / `</think>` pair.

## Related default-semantics question

The same native GLM command without `--enable-thinking` still emits the complete
`<think>...</think><answer>...` response. That may be intended for an explicitly
named Thinking checkpoint. If so, documenting that `enable_thinking=false`
cannot disable reasoning for every checkpoint would help callers distinguish
template control from model behaviour.
