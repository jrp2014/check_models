# Draft upstream issue: thinking_budget cannot bound models that emit their own thinking start token

Status: **posted as [Blaizzy/mlx-vlm#1819](https://github.com/Blaizzy/mlx-vlm/issues/1819)**;
fix PR [#1882](https://github.com/Blaizzy/mlx-vlm/pull/1882) open, unmerged. This
file tracks the evidence behind the issue and its follow-up comments. Native-run
evidence below was captured 2026-08-15 (`8683ec19`) and re-verified 2026-08-17
(`625f71fa`, see the update at the end); sweep evidence from the 2026-08-14 run.

---

## `thinking_budget` is silently ignored when the model emits its own thinking start token (GLM-4.1V-Thinking, Kimi-VL-Thinking)

### Summary

`ThinkingBudgetCriteria` only engages when the thinking start token is already
present in the **prompt** (i.e. the chat template pre-opens `<think>`). Models
whose templates do *not* pre-open the block, but which emit the start token as
their first generated token — GLM-4.1V-9B-Thinking emits `<think>`,
Kimi-VL-A3B-Thinking emits `◁think▷` — can never be budgeted: the same
`--thinking-budget` that visibly bounds Qwen3-VL-Thinking does nothing for
them, and on longer tasks they reason until `max_tokens` so the requested
answer is truncated away.

Verified at current `main` (`8683ec19`) and in the `v0.6.13` release (same
code at `mlx_vlm/generate/dispatch.py:917`).

### Where the gate closes

1. `generate()` ANDs `enable_thinking` with prompt containment
   (`mlx_vlm/generate/dispatch.py:913-921`):

   ```python
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

Generate a small test image (any image works; this one is regenerable):

```python
from PIL import Image, ImageDraw
img = Image.new("RGB", (512, 384), (70, 130, 180))
draw = ImageDraw.Draw(img)
draw.rectangle([40, 250, 470, 350], fill=(240, 220, 130))
draw.ellipse([380, 40, 460, 120], fill=(255, 215, 0))
draw.polygon([(120, 250), (180, 160), (240, 250)], fill=(178, 34, 34))
img.save("repro.png")
```

Run the same command against both models with a deliberately tight budget:

```bash
python -m mlx_vlm.generate \
  --model mlx-community/GLM-4.1V-9B-Thinking-8bit \
  --revision 9677807f106500eb7690391c27645d59f6855cfb \
  --image repro.png \
  --prompt "Give a one-sentence description." \
  --max-tokens 300 --temperature 0.0 \
  --enable-thinking --thinking-budget 20
```

```bash
python -m mlx_vlm.generate \
  --model mlx-community/Qwen3-VL-2B-Thinking-bf16 \
  --revision c325e5ea14c215bb08fa0d668c81fa2581f9050b \
  --image repro.png \
  --prompt "Give a one-sentence description." \
  --max-tokens 300 --temperature 0.0 \
  --enable-thinking --thinking-budget 20
```

### Observed output (macOS 26.6.1, M5 Max, mlx 0.32.1.dev20260814+3d23f7d87)

**Qwen3-VL-2B-Thinking** (template pre-opens `<think>` — budget works):
thinking is force-closed mid-sentence at the 20-token budget and generation
continues outside the block:

```text
Got it, let's see. The image has a blue background, a yellow sun in the top right
</think>

A blue background features a yellow sun in the top right corner, a red triangle
partially above a light yellow rectangle.
```

**GLM-4.1V-9B-Thinking** (emits `<think>` itself — budget silently ignored):
the thinking block runs ~75 tokens, far past the 20-token budget, and closes
only when the model chooses to:

```text
<think>Got it, let's look at the image. There's a blue background, a yellow
rectangle (maybe a ground or platform), a red triangle (like a hill or shape),
and a yellow circle (like a sun). So the description should include the main
elements: a blue sky with a yellow sun, a red triangle on a yellow rectangle.
Let's make a one-sentence description.</think><answer>The image shows a blue
background with a yellow sun, a red triangle on a yellow rectangular platform.
```

On this trivial image GLM recovers because its reasoning is naturally short;
on real tasks the unbounded block routinely consumes the entire token budget.
In a 41-model catalogue sweep (max_tokens=1000, temperature 0.0), both
GLM-4.1V-9B-Thinking and Kimi-VL-A3B-Thinking spent all 1000 tokens inside
their self-opened thinking blocks and produced no answer, while the
template-opened thinkers (Qwen3-VL-Thinking, ERNIE-4.5-VL-Thinking) are
bounded by the same flag. Note the budget mechanism guarantees only closure
of the block; whether the remaining tokens suffice for a complete answer
still depends on `max_tokens` headroom.

### Sweep evidence (which models can be budgeted today)

| Model | Start token in rendered prompt? | Start token emitted in output? | Budgetable today? |
| --- | --- | --- | --- |
| Qwen3-VL-2B-Thinking-bf16 | yes (`<think>`) | — | yes (verified above) |
| ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | yes (`<think>`) | — | yes |
| GLM-4.1V-9B-Thinking-8bit | **no** | **yes** (`<think>`) | **no** (verified above) |
| Kimi-VL-A3B-Thinking-2506-bf16 | **no** | **yes** (`◁think▷`) | **no** |
| MiniCPM-V-4.6-8bit | closed stub (`<think>\n\n</think>`) | self-closes | n/a |

(Kimi-VL needs `--thinking-start-token "◁think▷" --thinking-end-token
"◁/think▷"`. Apriel-1.5-Thinker and X-Reasoner reason free-form with no
marker at all — out of scope for a token-based budget.)

### Suggested fix

Let the criteria arm itself on an **emitted** start token instead of relying
on the prompt-containment AND: in `ThinkingBudgetCriteria.__call__`, detect
`token_id == self.thinking_start_token_id` unconditionally (or gate it on the
user's original `enable_thinking` rather than the ANDed value) and set
`in_thinking = True` from that point, keeping the existing prompt-containment
path for templates that pre-open the block. That makes `--thinking-budget`
uniformly effective for both template-opened and self-opened thinking models,
with no behaviour change for models that never emit the token.

### Environment

- mlx-vlm: `main` @ `8683ec19` (editable install); also inspected `v0.6.13`
- mlx: `0.32.1.dev20260814+3d23f7d87`
- macOS 26.6.1, Apple M5 Max (128 GB), Python 3.13.14

---

## Update 2026-08-17 at `625f71fa`: the budget now fires, but the model resumes reasoning past it

Re-ran the GLM-4.1V-9B-Thinking case on current `main` (`625f71fa`, mlx
`0.32.1.dev20260817`, M5 Max), single image, `temperature=0.0`,
`max_tokens=300`, `enable_thinking=True`, `thinking_budget=20`. The rendered
prompt ends `…<|assistant|>\n` — the template does not open a thinking block;
the model emits `<think>` itself (the case #1882 targets).

Observed (229 tokens generated, `finish_reason=stop`):

1. The model opens `<think>` itself and the budget **does** engage: `</think>`
   is forced after **23 tokens**. This differs from the original report at
   0.6.11, where the block ran unclosed to the cap.
2. The model then **continues reasoning immediately without re-opening
   `<think>`** — `</think>Got it, let's look at the image. …` — for a further
   **204 tokens** of unmarked chain-of-thought.
3. It finally emits its *own* `</think>` (a second one — the model's protocol)
   followed by `<answer>A monument with a bird statue atop stands by the sea…</answer>`.

So the forced closure is treated as noise: the model neither stops thinking
nor transitions to answer mode. Net effect: the budget saved 0 tokens versus
no budget on this prompt; on the catalogue prompt in the sweeps the same
model still spends the full 1000-token cap.

Implications for #1882:

- The `input_ids` gate at `generate/dispatch.py:918` is unchanged; the budget
  engaged here only because `enable_thinking=True` was passed explicitly and
  the criteria armed on the emitted opener. A caller relying on auto-detection
  still gets nothing.
- Closing the block is **necessary but not sufficient**. For a self-opening
  model the forced `</think>` must be followed by something that moves the
  model into answer mode — e.g. forcing the model's answer-start token where
  one exists (`<answer>` for GLM-4.1V/4.6V), or at minimum resetting
  `in_thinking` and re-forcing closure if reasoning continues. Otherwise the
  model treats `\n</think>` as a stray token and carries on.

Retest command once #1882 is rebased: the same call with `thinking_budget=20`
should yield an `<answer>` within ~30–40 tokens if the fix achieves its goal.

Related, resolved: the Idefics3 `<end_of_utterance>` leak (same
turn-terminator class) was fixed upstream in `625f71fa` (#1936; see also the
earlier PR #1774), verified live. Kimi-VL's `<|im_assistant|>` leak has a
different cause — `generation_config.json` (`[163585]`) overriding a correct
`config.json` `eos_token_id` (`[163584, 163586]`) at load — and is tracked as
a follow-up comment on #1936.
