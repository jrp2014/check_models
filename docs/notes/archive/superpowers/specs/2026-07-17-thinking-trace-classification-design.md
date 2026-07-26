# Thinking Trace Classification Design

## Goal

Report expected thinking-model output accurately without treating the mere
presence of a documented thinking delimiter as a model or integration fault.
Continue to identify runs that fail to reach a usable final answer within the
configured generation budget.

## Classification

The quality analysis will distinguish three cases:

1. A documented thinking delimiter from a known thinking model is an
   informational `thinking-trace` signal.
2. A thinking block that starts but does not close is a `thinking-incomplete`
   quality signal. It means the run produced no separable final answer; it does
   not assign blame to mlx-vlm, Transformers, or the model.
3. Reasoning markers or chain-of-thought phrases from a model that is not known
   to be a thinking model remain a `reasoning-leak` signal.

Thinking-model recognition will reuse information already available for each
result, beginning with the model identifier. Kimi's documented `◁think▷` and
`◁/think▷` protocol will be recognised alongside the existing generic
`<think>` and `</think>` delimiters. The detector will not remove or rewrite
raw generated text.

## Verdicts and Ownership

An informational thinking trace will not, by itself, change the review verdict,
user recommendation bucket, or suspected owner. An incomplete thinking block
will contribute diagnostic evidence, while existing contract and cutoff logic
will continue to determine whether the result is degraded or unsuitable for
the configured captioning budget.

Consequently, the current Kimi results remain `cutoff_degraded` and `avoid`
because they consumed all 500 tokens without reaching the requested catalogue
answer. Their reports will no longer describe the expected opening delimiter
as a fault or as a generic reasoning leak.

## Reports

Markdown, HTML, TSV, and JSONL output will use human-readable wording that
separates `thinking trace present` from `thinking incomplete`. Raw thinking
text remains visible for reproducibility and model evaluation.

Developer-oriented escalation will treat incomplete thinking as actionable
benchmark evidence, but a complete expected trace will be informational only.

## Testing

Regression tests will cover:

- a known thinking model with a complete documented trace;
- a known thinking model with an unclosed trace;
- a non-thinking model emitting a reasoning marker;
- unchanged cutoff verdict and recommendation behaviour;
- report and compact-quality labels for the new signals.

The focused quality/report tests will run before the repository's formatting,
lint, and full quality gates.
