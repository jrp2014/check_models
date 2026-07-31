# Paste-Ready Run Issue Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a compact aggregate GitHub issue body for a model-matrix run and regenerate it from retained JSONL/run data without model inference.

**Architecture:** Add one JSONL-backed Markdown renderer to the intentional `src/check_models.py` monolith. Normal finalization and report-only regeneration both call the same renderer, which consumes cached schema `2.0` assessments verbatim, expands actionable failures, links other surfaced rows to full diagnostics, and never embeds heavyweight evidence.

**Tech Stack:** Python 3.13, dataclasses and TypedDicts, existing report-block Markdown renderer, pytest, Ruff, markdownlint-cli2.

## Global Constraints

- Write `output/issues/run_summary.md` only when the retained run has an actionable failure, an observation requiring reproduction, or an indeterminate attempt.
- Never reclassify retained results; serialized JSONL `assessment` values are authoritative.
- Never embed full tracebacks, generated output, the complete prompt, Python reproduction scripts, or the full environment inventory.
- List every surfaced non-actionable model in canonical result order; count clean completions without listing them.
- Use repository GitHub links for default production output and relative links for temporary/custom outputs.
- Preserve all retained input artifacts during report-only regeneration.
- Run every Python and Make command inside the `mlx-vlm` conda environment.

---

### Task 1: JSONL-backed compact issue renderer

**Files:**

- Modify: `src/check_models.py` in the report block types and diagnostics/report-generator sections
- Test: `src/tests/test_report_generation.py`

**Interfaces:**

- Produces `RunIssueSummarySource`, a frozen dataclass containing validated metadata, cached result records, optional image/settings, versions, and system facts.
- Produces `_load_run_issue_summary_source(results_jsonl: Path, run_json: Path | None) -> RunIssueSummarySource`.
- Produces `generate_run_issue_summary_report(output_paths: ReportOutputPaths, *, issue_reports: Mapping[str, Path] | None = None) -> Path | None`.
- Extends report table cells with a safe explicit-target link type so evidence cells render external or relative artifact links without raw Markdown.
- Produces test-local `_issue_summary_output_paths(output_dir: Path) -> ReportOutputPaths`, `_issue_summary_result(model: str, *, execution: str = "completed", usability: str = "usable", maintainer_status: str = "none", observations: list[str] | None = None) -> dict[str, object]`, and `_write_issue_summary_fixture(output_paths: ReportOutputPaths, *, results: Sequence[dict[str, object]]) -> None` helpers with literal schema `2.0` data; these helpers never call production serialization code.

- [ ] **Step 1: Write a failing mixed-run rendering test**

Create literal schema `2.0` JSONL metadata/result records for one crash, one observed completion, and one clean completion. Write a minimal run JSON record, invoke the wished-for renderer under relative-link mode, and assert consumer-visible behavior:

```python
def test_run_issue_summary_expands_crash_and_tables_other_findings(tmp_path: Path) -> None:
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(
        output_paths,
        results=(
            _issue_summary_result("org/crash", execution="crashed"),
            _issue_summary_result(
                "org/observed",
                usability="unusable",
                maintainer_status="observation_needs_reproduction",
                observations=["repeated_output"],
            ),
            _issue_summary_result("org/clean"),
        ),
    )

    with patch.object(check_models._LinkStyleState, "value", "relative"):
        summary = check_models.generate_run_issue_summary_report(output_paths)

    assert summary == output_paths.index.parent / "issues" / "run_summary.md"
    content = summary.read_text(encoding="utf-8")
    assert "## Actionable failures" in content
    assert "### org/crash" in content
    assert "processor_load" in content
    assert "ValueError: processor missing image support" in content
    assert "python reproduce.py org/crash" in content
    assert "| org/observed | completed / unusable | repeated output |" in content
    assert "../reports/diagnostics.md#diagnostic-org-observed" in content
    assert "1 clean completion" in content
    assert "org/clean" not in content
    assert "Traceback (most recent call last)" not in content
    assert "generated output that must not be copied" not in content
    assert "full prompt that must not be copied" not in content
```

The production mutation this catches is rendering heavyweight evidence, losing the crash/table distinction, reordering surfaced rows, or omitting evidence links.

- [ ] **Step 2: Run the mixed-run test and verify RED**

Run:

```bash
pytest src/tests/test_report_generation.py::test_run_issue_summary_expands_crash_and_tables_other_findings -q
```

Expected: FAIL because `generate_run_issue_summary_report` does not exist.

- [ ] **Step 3: Implement validated retained-input loading and compact rendering**

Add a frozen source record and a single renderer. Validate `_type`, format version, assessment enum values, observation strings, failure mappings, and model provenance before constructing the source. Read run JSON only for optional image and common settings; preserve JSONL metadata as the fallback.

Render these blocks in order:

```text
# mlx-vlm compatibility findings across N cached vision-language models
## Run summary
## Actionable failures
### MODEL
## Other surfaced results
| Model | Execution / usability | Observations | Full evidence |
## Clean completions
## Run context
## Full artifacts
```

Use `_diagnostics_model_anchor()` and `_markdown_artifact_target()` for evidence targets. Render only the root exception-chain lines from `failure.exception_chain`; never use `failure.traceback`, `generated_text`, `captured_output_on_fail`, or the metadata prompt. Build the one-line reproduction as:

```text
python reproduce.py MODEL --revision REVISION --image IMAGE --prompt-file prompt.txt
```

Omit absent optional pairs. Return `None` and unlink a stale summary when no retained assessment is surfaced.

- [ ] **Step 4: Run the mixed-run test and verify GREEN**

Run the same focused pytest command and confirm PASS.

- [ ] **Step 5: Write failing retained-contract tests**

Add independent tests proving that the renderer:

```python
def test_run_issue_summary_preserves_cached_assessment_without_reclassification(
    tmp_path: Path,
) -> None:
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    cached = _issue_summary_result(
        "org/cached",
        usability="usable_with_caveats",
        maintainer_status="observation_needs_reproduction",
        observations=["draft_returned_unchanged"],
    )
    _write_issue_summary_fixture(output_paths, results=(cached,))
    with patch.object(check_models._LinkStyleState, "value", "relative"):
        summary = check_models.generate_run_issue_summary_report(output_paths)
    assert summary is not None
    content = summary.read_text(encoding="utf-8")
    assert "completed / usable with caveats" in content
    assert "draft returned unchanged" in content

def test_run_issue_summary_clean_run_removes_stale_summary(tmp_path: Path) -> None:
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(
        output_paths,
        results=(_issue_summary_result("org/clean"),),
    )
    stale = output_paths.index.parent / "issues" / "run_summary.md"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale", encoding="utf-8")
    generated = check_models.generate_run_issue_summary_report(output_paths)
    assert generated is None
    assert not stale.exists()

@pytest.mark.parametrize("bad_line", ["not-json", '{"_type":"result"}'])
def test_run_issue_summary_rejects_malformed_jsonl_without_rewriting_sources(
    tmp_path: Path,
    bad_line: str,
) -> None:
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(output_paths, results=())
    output_paths.jsonl.write_text(bad_line + "\n", encoding="utf-8")
    original_jsonl = output_paths.jsonl.read_bytes()
    original_run_json = output_paths.run_json.read_bytes()
    with pytest.raises(ValueError):
        check_models.generate_run_issue_summary_report(output_paths)
    assert output_paths.jsonl.read_bytes() == original_jsonl
    assert output_paths.run_json.read_bytes() == original_run_json
```

The production mutations caught are calling `_assess_result`, leaving stale issue bodies, silently accepting invalid evidence, or modifying canonical retained inputs.

- [ ] **Step 6: Run the retained-contract tests and verify RED**

Run:

```bash
pytest src/tests/test_report_generation.py -k "run_issue_summary" -q
```

Expected: at least the malformed/stale/cached-assessment cases fail until validation and cleanup are complete.

- [ ] **Step 7: Complete minimal validation and stale cleanup**

Implement only the validation branches demanded by the failing tests. A missing JSONL file raises `OSError`; malformed JSON, a missing metadata row, an unsupported format version, or malformed assessment raises `ValueError`. Missing or malformed run JSON is ignored as optional enrichment.

- [ ] **Step 8: Run all run-summary tests and verify GREEN**

Run:

```bash
pytest src/tests/test_report_generation.py -k "run_issue_summary" -q
```

Expected: all run-summary tests pass.

- [ ] **Step 9: Commit the renderer**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "feat: add paste-ready run issue summary"
```

### Task 2: Finalization and output-index integration

**Files:**

- Modify: `src/check_models.py` in `DiagnosticsArtifacts`, report orchestration, index generation, and maintainer logging
- Test: `src/tests/test_report_generation.py`
- Test: `src/tests/test_jsonl_output.py`

**Interfaces:**

- Extends `DiagnosticsArtifacts` with `run_issue_summary: Path | None = None`.
- Extends `generate_output_index_report(filename: Path, *, output_paths: ReportOutputPaths, issue_reports: Mapping[str, Path] | None = None, run_issue_summary: Path | None = None) -> None`.
- Normal report finalization calls `generate_run_issue_summary_report()` after JSONL, diagnostics/crash drafts, and run JSON are available, before it writes the final index.

- [ ] **Step 1: Write failing index and orchestration tests**

Extend the existing output-index test with a real summary path and crash mapping:

```python
check_models.generate_output_index_report(
    output_paths.index,
    output_paths=output_paths,
    run_issue_summary=output_dir / "issues" / "run_summary.md",
    issue_reports={"org/crash": output_dir / "issues" / "issue_org_crash.md"},
)
assert content.index("[Run issue summary]") < content.index("[org/crash]")
```

Extend the orchestration test so its temp output receives `issues/run_summary.md`, the index links it, and the returned artifact outcomes contain an isolated `run_issue_summary` success without rewriting tracked output.

- [ ] **Step 2: Run the integration tests and verify RED**

Run:

```bash
pytest src/tests/test_report_generation.py -k "output_index or report_orchestration" -q
```

Expected: FAIL because the index and orchestration do not yet accept or generate the aggregate summary.

- [ ] **Step 3: Wire summary generation into finalization**

Generate JSONL first, retain the current isolated diagnostics/crash-draft step, generate run JSON, then invoke the aggregate summary renderer in its own `try/except (OSError, TypeError, ValueError, RuntimeError)` block. Record `ReportArtifactOutcome(key="run_issue_summary", path=summary_path, succeeded=True)` only when a file is generated. Pass the resulting path to the index and log it as `Run Issue Summary`.

Keep HTML/gallery generation independent. A summary failure must still allow index, diagnostics, JSONL, run JSON, HTML, and gallery generation to proceed.

- [ ] **Step 4: Run the integration tests and verify GREEN**

Run the same focused pytest selection and confirm PASS.

- [ ] **Step 5: Run the complete report/JSONL regression set**

Run:

```bash
pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_markdown_formatting.py -q
```

Expected: all tests pass and all generated paths remain under `tmp_path`.

- [ ] **Step 6: Commit integration**

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py
git commit -m "feat: publish aggregate run issue summary"
```

### Task 3: Documentation, retained-output regeneration, and verification

**Files:**

- Modify: `src/README.md`
- Modify: `CHANGELOG.md`
- Create: `src/output/issues/run_summary.md`
- Modify: `src/output/index.md`
- Modify: `src/output/run.json` only if orchestration intentionally records the new artifact path
- Test: `src/tests/test_markdown_formatting.py`

**Interfaces:**

- Produces `regenerate_run_issue_summary(output_dir: Path) -> Path | None`, a canonical-path wrapper around the tested JSONL-backed renderer.
- Consumes existing `src/output/results.jsonl`, optional `src/output/run.json`, current crash drafts, and companion artifact paths.

- [ ] **Step 1: Write a failing report-only regeneration test**

Create a temp canonical output tree, call the wrapper, and assert that only `issues/run_summary.md` changes:

```python
before = {
    path: path.read_bytes()
    for path in (output_dir / "results.jsonl", output_dir / "run.json")
}
generated = check_models.regenerate_run_issue_summary(output_dir)
assert generated == output_dir / "issues" / "run_summary.md"
assert {path: path.read_bytes() for path in before} == before
```

The production mutation caught is report-only regeneration rewriting or requiring live inference artifacts.

- [ ] **Step 2: Run the regeneration test and verify RED**

Run:

```bash
pytest src/tests/test_report_generation.py -k "regenerate_run_issue_summary" -q
```

Expected: FAIL because the canonical-path wrapper does not exist.

- [ ] **Step 3: Implement the canonical-path wrapper**

Construct `ReportOutputPaths` rooted at the supplied directory, discover existing `issue_*.md` crash drafts using the same filename normalization as `_generate_github_issue_reports`, and delegate to `generate_run_issue_summary_report`. Do not add a second renderer or parse diagnostics Markdown.

- [ ] **Step 4: Run the regeneration test and verify GREEN**

Run the same focused pytest command and confirm PASS.

- [ ] **Step 5: Update documentation and changelog**

Document `issues/run_summary.md` in the README artifact list and output tree as the compact whole-run issue body; keep diagnostics documented as the complete evidence. Add an `[Unreleased]` changelog entry covering normal generation, report-only regeneration, and the index link.

- [ ] **Step 6: Regenerate the current retained artifact without inference**

Run inside the activated environment:

```bash
PYTHONPATH=src python -c 'from pathlib import Path; from check_models import regenerate_run_issue_summary; print(regenerate_run_issue_summary(Path("src/output")))'
```

Expected: prints `src/output/issues/run_summary.md` (or its resolved equivalent) and does not invoke model discovery or generation.

- [ ] **Step 7: Inspect the generated artifact**

Confirm:

- the one current crash is expanded outside the table;
- all other surfaced models appear exactly once in canonical order;
- clean models appear only as a count;
- links target the complete retained artifacts;
- the file contains no full traceback, complete model output, full prompt, or Python script;
- the representative artifact stays near the 100–150-line target and is practical to paste wholesale.

- [ ] **Step 8: Run formatting and focused preflight**

Run:

```bash
make format
make -C src lint-fix
make lint
pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_markdown_formatting.py -q
src/node_modules/.bin/markdownlint-cli2 src/output/issues/run_summary.md src/output/index.md
```

Expected: every command exits zero.

- [ ] **Step 9: Run the full quality gate**

Run:

```bash
make quality
```

Expected: all repository quality checks and the full pytest suite pass.

- [ ] **Step 10: Verify the final diff and commit**

Run `git diff --check`, inspect `git status --short`, and verify that no retained artifact other than the intentional summary/index (and optional run manifest) changed. Then commit:

```bash
git add CHANGELOG.md src/README.md src/output/issues/run_summary.md src/output/index.md src/check_models.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_markdown_formatting.py
git commit -m "docs: generate current run issue summary"
```
