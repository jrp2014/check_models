# Draft upstream issue: thinking_budget cannot bound models that emit their own thinking start token

Status: draft, not filed. Paste-ready body below the line. Evidence from the
2026-08-14 41-model sweep (`src/output/results.jsonl`, max_tokens=1000,
temperature 0.0, image cataloguing prompt).

---

## `thinking_budget` never engages for models that emit the thinking start token during generation (e.g. GLM-4.1V-Thinking, Kimi-VL-Thinking)

### Summary

`ThinkingBudgetCriteria` can only bound a thinking block whose start token is
already present in the **prompt** (i.e. the chat template pre-opens
`<think>`). Models whose templates do *not* pre-open the block, but which emit
the start token as their first generated token — GLM-4.1V-9B-Thinking emits
`<think>`, Kimi-VL-A3B-Thinking emits `◁think▷` — can never be budgeted: they
reason until `max_tokens` and the requested answer is truncated away, no
matter what `--thinking-budget` is set to.

Verified at current `main` (`8683ec19`) and in the `v0.6.13` release (same
code at `mlx_vlm/generate/dispatch.py:917`).

### Where the gate closes

1. `generate()` ANDs `enable_thinking` with prompt containment
   (`mlx_vlm/generate/dispatch.py:913-921`):

   ```python
   if thinking_budget is not None:
       thinking_start_token_id = tokenizer.encode(
           thinking_start_token, add_special_tokens=False
       )[-1]
       enable_thinking = enable_thinking and (
           thinking_start_token_id in input_ids.flatten().tolist()
       )
   ```

   If the rendered prompt lacks the start token, the criteria is constructed
   with `enable_thinking=False`.

2. `ThinkingBudgetCriteria.__call__` (`mlx_vlm/utils.py`) only recognises an
   *emitted* start token when `enable_thinking` is truthy:

   ```python
   if self.enable_thinking and token_id == self.thinking_start_token_id:
       self.in_thinking = True
   ```

   With `enable_thinking` forced to `False` by step 1, `in_thinking` starts
   `False` and can never become `True`, so `thinking_token_count` never
   increments and the budget is structurally inert for these models.

### Reproduction

Any image works; the model is the variable.

```bash
python -m mlx_vlm.generate \
  --model mlx-community/GLM-4.1V-9B-Thinking-8bit \
  --image any-local-image.jpg \
  --prompt "Give a one-sentence description." \
  --max-tokens 300 \
  --temperature 0.0 \
  --enable-thinking \
  --thinking-budget 100
```

**Expected:** after ~100 thinking tokens, the forced `\n</think>` sequence
closes the block and the model produces the answer within 300 tokens.

**Actual:** the rendered GLM-4.1V prompt contains no `<think>`, so the budget
is disabled by the prompt-containment AND; the model emits `<think>` as its
first generated token, reasons for all 300 tokens, and the output is cut off
mid-reasoning with no answer. Same behaviour for
`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` with
`--thinking-start-token "◁think▷" --thinking-end-token "◁/think▷"`.

### Evidence from a 41-model sweep (max_tokens=1000, temperature 0.0)

| Model | Start token in rendered prompt? | Start token emitted in output? | Budgetable today? | Outcome at cap |
| --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Thinking-bf16 | yes (`<think>`) | — | yes | truncated (budget not set in this run) |
| ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | yes (`<think>`) | — | yes | truncated (budget not set in this run) |
| GLM-4.1V-9B-Thinking-8bit | **no** | **yes** (`<think>`) | **no** | truncated, no answer |
| Kimi-VL-A3B-Thinking-2506-bf16 | **no** | **yes** (`◁think▷`) | **no** | truncated, no answer |
| MiniCPM-V-4.6-8bit | yes (`<think>`) | yes (self-closes) | yes | completed |

(Apriel-1.5-Thinker and X-Reasoner also burn the cap on free-form reasoning,
but emit no recognisable marker at all — out of scope for a token-based
budget.)

### Suggested fix

Let the criteria arm itself on an **emitted** start token instead of relying
on the prompt-containment AND:

- In `ThinkingBudgetCriteria.__call__`, detect
  `token_id == self.thinking_start_token_id` unconditionally (or gate it on
  the user's original `enable_thinking` rather than the ANDed value) and set
  `in_thinking = True` from that point.
- Keep the existing prompt-containment path for templates that pre-open the
  block (`in_thinking = True` from token 0).

That makes `--thinking-budget` uniformly effective for both template-opened
and self-opened thinking models, with no behaviour change for models that
never emit the token.

### Environment

- mlx-vlm: `main` @ `8683ec19` (editable install); also inspected `v0.6.13`
- mlx: `0.32.1.dev20260814+3d23f7d87`
- macOS 26.6.1, Apple M5 Max (128 GB), Python 3.13.14
