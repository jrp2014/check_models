# Thinking Trace Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate expected thinking-model delimiters from unexpected reasoning
leakage while reporting an unclosed thinking block as incomplete output.

**Architecture:** Extend the existing quality-analysis signals in
`src/check_models.py`; do not introduce another reporting pipeline. Pass the
model identifier into analysis where it is already available, classify paired
thinking delimiters once, and let all existing JSONL, TSV, Markdown, HTML, and
console consumers reuse the resulting dataclass fields and quality labels.

**Tech Stack:** Python 3.13, dataclasses, pytest, Ruff, mypy, ty, pyrefly,
Markdownlint.

## Global Constraints

- Keep `src/check_models.py` as the intentional single-file monolith.
- Preserve raw generated text without removing thinking content or delimiters.
- A complete expected trace is informational and does not change verdict,
  recommendation bucket, ownership, or maintainer escalation.
- An unclosed expected trace is diagnostic evidence; existing cutoff and output
  contract rules continue to determine the verdict and recommendation.
- Do not suppress lint or typing rules.
- Run every Python and Make command in the `mlx-vlm` Conda environment.

---

### Task 1: Classify expected and incomplete thinking traces

**Files:**

- Modify: `src/tests/test_quality_analysis.py`
- Modify: `src/check_models.py`

**Interfaces:**

- Consumes: `analyze_generation_text(text, generated_tokens, ..., model_name=None)`
- Produces: `GenerationQualityAnalysis.has_thinking_trace: bool`,
  `thinking_trace_incomplete: bool`, and `thinking_trace_markers: list[str]`
- Preserves: `has_reasoning_leak` and `reasoning_leak_markers` for unexpected
  reasoning output

- [ ] **Step 1: Write failing detector tests**

Add tests which call the public analysis function with a Kimi Thinking model
identifier and assert that a complete `◁think▷...◁/think▷` trace sets
`has_thinking_trace` without `has_reasoning_leak`, while an unclosed trace also
sets `thinking_trace_incomplete`. Retain the existing non-thinking `<think>`
test and explicitly assert it remains a reasoning leak.

```python
def test_analyze_generation_text_treats_expected_thinking_trace_as_information() -> None:
    text = "◁think▷Inspect the scene.◁/think▷ Title: Two boats on a river"
    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=24,
        model_name="mlx-community/Kimi-VL-A3B-Thinking-2506-bf16",
    )
    assert analysis.has_thinking_trace is True
    assert analysis.thinking_trace_incomplete is False
    assert analysis.has_reasoning_leak is False


def test_analyze_generation_text_flags_unclosed_expected_thinking_trace() -> None:
    prompt = (
        "Return exactly these three sections:\n"
        "Title: 5-10 words.\nDescription: 1-2 sentences.\nKeywords: 10-18 terms."
    )
    analysis = check_models.analyze_generation_text(
        "◁think▷Inspecting the image step by step.",
        generated_tokens=500,
        prompt=prompt,
        requested_max_tokens=500,
        model_name="mlx-community/Kimi-VL-A3B-Thinking-8bit",
    )
    assert analysis.has_thinking_trace is True
    assert analysis.thinking_trace_incomplete is True
    assert analysis.has_reasoning_leak is False
    assert analysis.verdict == "cutoff_degraded"
    assert analysis.user_bucket == "avoid"
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && pytest src/tests/test_quality_analysis.py -k "thinking_trace or reasoning_leak" -q'
```

Expected: the new tests fail because `model_name` and the thinking-trace fields
do not yet exist; the existing reasoning-leak test passes.

- [ ] **Step 3: Implement minimal classification**

Add immutable delimiter pairs for `<think>...</think>` and
`◁think▷...◁/think▷`. Add a compact signal dataclass or equivalent typed return
value, classify a delimiter as expected only when the model identifier contains
`thinking`, and route the fields through `PromptQualitySignals` and
`GenerationQualityAnalysis`. Add `model_name: str | None = None` to
`analyze_generation_text()` and `_analyze_text_quality()`, then pass
`result.model_name` and `params.model_identifier` at the successful and captured
failure analysis call sites.

Do not count an informational trace in `has_any_issues()`. Count
`thinking_trace_incomplete`, add `thinking_incomplete` to review evidence, and
leave `_classify_review_verdict()` unchanged so its cutoff/contract evidence
remains authoritative.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit the detector behaviour**

```bash
git add src/check_models.py src/tests/test_quality_analysis.py
git commit -m "fix: distinguish expected thinking traces"
```

### Task 2: Align report and machine-readable labels

**Files:**

- Modify: `src/tests/test_quality_analysis.py`
- Modify: `src/tests/test_report_generation.py`
- Modify: `src/check_models.py`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes: the three thinking-trace fields added by Task 1
- Produces: `thinking-trace` informational and `thinking-incomplete` diagnostic
  quality labels plus human-readable report summaries

- [ ] **Step 1: Write failing label/report tests**

Extend the compact quality-log test with expected values for
`thinking_trace=True` and `thinking_incomplete=True`. Add a report-focused test
that verifies the human signal text says `Thinking trace incomplete` and does
not say `Reasoning leak` for a known Kimi Thinking result.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && pytest src/tests/test_quality_analysis.py src/tests/test_report_generation.py -k "thinking or reasoning_leak" -q'
```

Expected: new exact-string/report assertions fail because the new labels are not
yet rendered.

- [ ] **Step 3: Implement report wording through existing helpers**

Add `Thinking trace present` and `Thinking trace incomplete` to
`GenerationQualityAnalysis.issues`, `_build_quality_issues_string()`, the compact
quality log, console warnings, review focus, evidence-label extraction, and the
existing `_summarize_quality_signals()` builder tuple. Only incomplete thinking
belongs in `_is_review_escalation()`; a complete trace remains visible in
quality details without entering the escalation/watchlist.

Add a concise `[Unreleased]` changelog entry describing the ownership-neutral
classification and unchanged cutoff behaviour.

- [ ] **Step 4: Run focused quality and report tests and verify GREEN**

Run the command from Step 2, then run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_quality_analysis.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py -q'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit report semantics**

```bash
git add src/check_models.py src/tests/test_quality_analysis.py src/tests/test_report_generation.py CHANGELOG.md
git commit -m "fix: report incomplete thinking without fault attribution"
```

### Task 3: Verify the repository gate

**Files:**

- Modify only if an automated formatter makes an intentional change.

**Interfaces:**

- Consumes: Tasks 1 and 2
- Produces: clean formatting, lint, static analysis, tests, shell checks, and
  Markdown lint

- [ ] **Step 1: Apply formatting and safe lint fixes**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make format && make -C src lint-fix'
```

- [ ] **Step 2: Run Ruff lint**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make lint'
```

Expected: exit 0 with no Ruff findings.

- [ ] **Step 3: Run commit hygiene**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && bash src/tools/run_commit_hygiene.sh'
```

Expected: exit 0.

- [ ] **Step 4: Run the full quality gate**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make quality'
```

Expected: exit 0 across typing, static analysis, the full pytest suite,
ShellCheck, and Markdownlint.

- [ ] **Step 5: Inspect final scope and commit any formatter-only changes**

```bash
git diff --check
git status --short
git diff --stat origin/main...HEAD
```

If formatting changed tracked files, stage only the files in this plan and
commit them with `style: format thinking trace classification`.
