# Issue-ready Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the model gallery preserve readable model formatting and make diagnostics a compact, self-contained mlx-vlm issue report without losing evidence for highlighted models.

**Architecture:** Extend the existing typed report-block representation with internal links and model-output presentation, then build one diagnostic block tree from the cached `ReportRenderContext` for both Markdown and HTML. Keep classification and canonical evidence untouched, move repeated reproduction context to one shared section, and delete the parallel format-specific diagnostic assembly.

**Tech Stack:** Python 3.13, dataclasses and PEP 695 typing, existing `tabulate`/HTML/Markdown helpers, pytest, Ruff, mypy, ty, pyrefly, Skylos, markdownlint-cli2.

## Global Constraints

- Always activate the `mlx-vlm` Conda environment before Python or Make commands.
- Run Conda-backed validation outside the sandbox if required to avoid the known rattler panic.
- Keep `src/check_models.py` as the intentional single-file monolith.
- Use the cached `ResultAssessment`; do not add classification, scoring, model-name, keyword, or image-specific heuristics.
- Preserve complete highlighted output, partial output, tracebacks, and captured streams without truncation.
- Keep raw model evidence inert and byte-for-byte exact; neutralise HTML and GitHub mentions only in the presentation view.
- Do not add a Python Markdown-rendering dependency.
- Keep all parameters and return values fully and narrowly typed; do not introduce `Any` or `object` where a closed union is possible.
- Do not add lint or type suppressions. Remove newly obsolete code and suppressions where possible.
- Validation tests must render into `tmp_path` or ignored test paths and must not modify tracked `src/output/` artefacts.
- Do not hand-edit or reconstruct the current retained run reports.
- Record `wc -l src/check_models.py` before and after implementation; the final count must be below the 16,214-line baseline without a brittle line-count unit test.
- Run `make format`, `make -C src lint-fix`, and `make lint` before `make quality`.
- Update `CHANGELOG.md` under `[Unreleased]` for the report and refactor changes.

---

### Task 1: Tighten and extend the shared report blocks

**Files:**

- Modify: `src/check_models.py:1892-2155`
- Test: `src/tests/test_markdown_formatting.py`

**Interfaces:**

- Consumes: existing `ReportParagraph`, `ReportKeyValues`, `ReportTable`, `ReportCodeBlock`, `ReportDetails`, `ReportRaw`, and `ReportSection` rendering.
- Produces: `ReportLink`, `ReportModelOutput`, `ReportCell`, and recursive `ReportBlock`; `render_report_markdown(blocks: Sequence[ReportBlock]) -> list[str]`; `render_report_html(blocks: Sequence[ReportBlock]) -> list[str]`.

- [ ] **Step 1: Add failing shared-renderer tests**

Add tests that exercise the format-specific parts through the public internal renderers:

```python
def test_shared_report_blocks_render_links_and_safe_model_output() -> None:
    captured = "## Heading\n\n- first\n- second\n\n@mlx-user <script>bad()</script>"
    blocks: tuple[check_models.ReportBlock, ...] = (
        check_models.ReportTable(
            headers=("Model",),
            rows=((check_models.ReportLink("org/model", "diagnostic-org-model"),),),
        ),
        check_models.ReportModelOutput(captured),
    )

    markdown = "\n".join(check_models.render_report_markdown(blocks))
    html_output = "\n".join(check_models.render_report_html(blocks))

    assert "[org/model](#diagnostic-org-model)" in markdown
    assert "> ## Heading" in markdown
    assert "> - first" in markdown
    assert r"\@mlx-user" in markdown
    assert "&lt;script&gt;bad()&lt;/script&gt;" in markdown
    assert "<summary>Exact raw output</summary>" in markdown
    assert captured in markdown
    assert '<a href="#diagnostic-org-model">org/model</a>' in html_output
    assert "&lt;script&gt;bad()&lt;/script&gt;" in html_output
```

Add a second test with nested backtick fences, tabs, and trailing spaces. Assert
that the complete source string occurs contiguously exactly once in Markdown,
inside the dynamically sized raw fence.

- [ ] **Step 2: Run the new tests and verify the expected failure**

Run:

```bash
pytest src/tests/test_markdown_formatting.py -k "shared_report_blocks or model_output" -q
```

Expected: collection or attribute failures because `ReportLink`,
`ReportModelOutput`, and `ReportBlock` do not exist.

- [ ] **Step 3: Implement narrow recursive report types**

Add immutable types alongside the existing report blocks:

```python
@dataclass(frozen=True)
class ReportLink:
    """Internal report link rendered safely in Markdown and HTML."""

    label: str
    anchor: str


@dataclass(frozen=True)
class ReportModelOutput:
    """Readable presentation plus exact raw captured model output."""

    content: str
    raw_summary: str = "Exact raw output"


type ReportCell = str | ReportLink
type ReportBlock = (
    ReportParagraph
    | ReportKeyValues
    | ReportBulletList
    | ReportTable
    | ReportCodeBlock
    | ReportDetails
    | ReportRaw
    | ReportSection
    | ReportModelOutput
)
```

Change `ReportTable.rows` to `tuple[tuple[ReportCell, ...], ...]` and the
recursive `ReportDetails.blocks` / `ReportSection.blocks` fields to
`tuple[ReportBlock, ...]`. Change every shared renderer parameter from `object`
to `ReportBlock` and every sequence from `Sequence[object]` to
`Sequence[ReportBlock]`.

Render `ReportLink` as an escaped Markdown internal link and an escaped HTML
anchor. Render `ReportModelOutput` as:

- Markdown: `Readable output`, HTML-escaped and mention-neutralised blockquote
  lines with two-space hard breaks, followed by a collapsed exact
  `ReportCodeBlock`;
- HTML: an escaped preformatted readable view followed by a collapsed escaped
  raw code block.

Use `html.escape(..., quote=False)` for presentation text and replace every
literal `@` with `\\@` only in the Markdown readable view. Do not apply this
transformation to `content` passed to `ReportCodeBlock`.

- [ ] **Step 4: Run focused rendering and type checks**

Run:

```bash
pytest src/tests/test_markdown_formatting.py -k "shared_report_blocks or model_output" -q
make typecheck
make ty
```

Expected: focused tests pass and both type checkers report no diagnostics.

- [ ] **Step 5: Commit the typed report primitives**

```bash
git add src/check_models.py src/tests/test_markdown_formatting.py
git commit -m "refactor: tighten shared report blocks"
```

---

### Task 2: Preserve gallery preview and complete-output formatting

**Files:**

- Modify: `src/check_models.py:6680-6975`
- Modify: `src/check_models.py:2380-2505`
- Modify: `src/check_models.py:8390-8465`
- Test: `src/tests/test_report_generation.py:2650-2735`
- Test: `src/tests/test_markdown_formatting.py`

**Interfaces:**

- Consumes: `ReportModelOutput`, `_generation_text_value`, `MARKDOWN_ESCAPER`, `_truncate_text_preview`, and existing gallery facts.
- Produces: `_collapse_preview_line_whitespace(text: str) -> str`; gallery chooser rows containing preserved newlines; Markdown and HTML complete entries rendered through `ReportModelOutput`.

- [ ] **Step 1: Add failing gallery preview and evidence tests**

Create a successful result whose output contains headings, blank lines, a list,
an HTML tag, an `@` mention, and a nested fence:

```python
formatted_output = (
    "## Title\n\n"
    "Two cats resting\n\n"
    "- pink sofa\n"
    "- remote control\n\n"
    "@maintainer <details>unsafe</details>\n"
    "```text\nnested\n```"
)
```

Assert in the generated gallery that:

- the chooser cell contains `## Title<br><br>Two cats resting`;
- the readable view contains blockquoted headings and list items;
- the readable view contains `\\@maintainer` and escaped `<details>`;
- `<summary>Exact raw output</summary>` exists;
- `formatted_output` occurs contiguously once in the raw fence;
- the existing preview character bound still applies.

Add equivalent HTML assertions that the output is escaped and remains inside
preformatted blocks.

- [ ] **Step 2: Run the gallery tests and verify failure**

Run:

```bash
pytest src/tests/test_report_generation.py -k "gallery and (preview or complete_output)" -q
pytest src/tests/test_markdown_formatting.py -k "gallery" -q
```

Expected: the preview assertion fails because `_gallery_row` currently calls
`_collapse_preview_whitespace`; readable/raw assertions fail because gallery
entries use only a literal code block.

- [ ] **Step 3: Preserve preview line boundaries**

Implement:

```python
def _collapse_preview_line_whitespace(text: str) -> str:
    """Collapse horizontal whitespace while preserving source line boundaries."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(" ".join(line.split()) for line in normalized.split("\n"))
```

Use it in `_gallery_row` before `_truncate_text_preview`. Leave
`MARKDOWN_ESCAPER.escape(row.output_preview)` at the table boundary so retained
newlines become `<br>` and pipes remain escaped.

- [ ] **Step 4: Route complete gallery output through the shared block**

In `_render_gallery_model`, replace the successful output's manual emphasis and
code-block assembly with `render_report_markdown((ReportModelOutput(output),))`.
Keep the explicit `empty output` branch.

In `_html_gallery_model`, replace the successful output's direct
`_html_code_block` construction with `render_report_html((ReportModelOutput(output),))`.
Do not change crash evidence ordering or the outer complete-evidence `<details>`.

- [ ] **Step 5: Run the full gallery/report test slice**

Run:

```bash
pytest src/tests/test_markdown_formatting.py src/tests/test_report_generation.py -k "gallery or model_output" -q
```

Expected: all selected tests pass, including existing no-shortening and nested
fence tests.

- [ ] **Step 6: Commit the gallery change**

```bash
git add src/check_models.py src/tests/test_markdown_formatting.py src/tests/test_report_generation.py
git commit -m "feat: preserve gallery output formatting"
```

---

### Task 3: Build one skim-first diagnostic structure for Markdown and HTML

**Files:**

- Modify: `src/check_models.py:7720-8135`
- Modify: `src/check_models.py:8490-8670`
- Test: `src/tests/test_report_generation.py:1030-1315`
- Test: `src/tests/test_report_generation.py:1580-1725`

**Interfaces:**

- Consumes: `ReportBlock`, `ReportLink`, `ReportModelOutput`, `ReportDetails`, `ReportRenderContext`, `_run_outcome_counts`, `_valid_generation_tps`, and existing provenance/metric helpers.
- Produces: `DiagnosticsPartitions`; `_partition_diagnostics(context: HtmlReportContext) -> DiagnosticsPartitions`; shared model and report evidence block builders.

- [ ] **Step 1: Add failing partition and presentation tests**

Extend the existing four-result diagnostic fixture to include two clean
completions with processor/provenance/performance facts. Assert that generated
Markdown and HTML contain:

- outcome, maintainer-status, usability, and observation counts;
- one triage row per crash, observation, or indeterminate attempt and no clean
  result in the triage table;
- internal links to stable diagnostic anchors;
- an expanded crash section with traceback before partial output;
- collapsed observation and indeterminate summaries naming their status;
- a collapsed `Clean completions` table containing model, short revision,
  processor basename, stop reason, prompt/generated tokens, valid generation
  TPS, and peak memory;
- no clean generated output in diagnostics;
- complete observed output in the collapsed evidence.

Use distinctive strings such as `CLEAN-OUTPUT-MUST-NOT-APPEAR` and
`OBSERVED-OUTPUT-MUST-APPEAR` so the assertions cannot pass accidentally.

- [ ] **Step 2: Run focused diagnostics tests and verify failure**

Run:

```bash
pytest src/tests/test_report_generation.py -k "diagnostics and (partition or clean or evidence_order)" -q
```

Expected: failures for missing count/triage/clean sections and uncollapsed
observation evidence.

- [ ] **Step 3: Introduce the narrow partition value**

Add:

```python
@dataclass(frozen=True)
class DiagnosticsPartitions:
    """Direct cached-assessment partitions for maintainer presentation."""

    actionable: tuple[PerformanceResult, ...]
    observations: tuple[PerformanceResult, ...]
    indeterminate: tuple[PerformanceResult, ...]
    clean: tuple[PerformanceResult, ...]
```

Return all four partitions from `_partition_diagnostics`. Define clean as a
completed assessment with maintainer status `none`; do not infer it from output
text.

- [ ] **Step 4: Build shared diagnostic blocks**

Implement these exact narrow interfaces:

- `_diagnostics_model_anchor(model_name: str) -> str`
- `_diagnostics_model_blocks(result: PerformanceResult, assessment: ResultAssessment, *, run_args: argparse.Namespace | None, model_provenance: ModelProvenanceRecord | None) -> tuple[ReportBlock, ...]`
- `_diagnostics_evidence_blocks(report_context: HtmlReportContext, *, prompt: str, image_path: Path | None, run_args: argparse.Namespace | None) -> tuple[ReportBlock, ...]`

The model block builder returns concise facts plus exact evidence. Use
`ReportModelOutput` only for completed model text. Use `ReportCodeBlock` for
tracebacks, partial output, and captured streams. Omit absent optional evidence
instead of emitting an `unavailable` code block.

The report builder creates the summary counts, triage table with `ReportLink`,
expanded actionable `ReportSection` blocks, collapsed observation and
indeterminate `ReportDetails`, and the collapsed clean table. Add
`revision_preview_chars: int = 12` to `FormattingThresholds` and use processor
class basenames to keep the clean table skimmable; retain full values in detailed
facts.

For this independently working commit, append the existing direct CLI and
single-model Python reproduction blocks to each highlighted aggregate entry
outside `_diagnostics_model_blocks`. Task 4 will replace only that repeated
aggregate material; individual crash issue drafts continue to use it.

- [ ] **Step 5: Replace parallel Markdown and HTML assembly**

Make `generate_diagnostics_report` render `_diagnostics_evidence_blocks` with
`render_report_markdown`. Make `_html_maintainer_diagnostics` render the same
blocks with `render_report_html` inside its section wrapper.

Delete `_diagnostics_partition_section`, `_html_diagnostics_entry`, and
`_html_diagnostics_partition`. Keep a temporary direct issue-draft wrapper only
if Task 4 still needs it; do not duplicate evidence assembly.

- [ ] **Step 6: Run diagnostics, HTML, and consistency tests**

Run:

```bash
pytest src/tests/test_report_generation.py -k "diagnostics or standalone_html or canonical_assessment" -q
pytest src/tests/test_html_formatting.py src/tests/test_markdown_formatting.py -q
```

Expected: all selected tests pass and the existing cached-assessment assertions
still prove that renderers never reclassify results.

- [ ] **Step 7: Commit the shared diagnostic structure**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "feat: make diagnostics skim-first"
```

---

### Task 4: Share reproduction and provenance once per diagnostics report

**Files:**

- Modify: `src/check_models.py:7780-8050`
- Modify: `src/check_models.py:14720-14865`
- Test: `src/tests/test_report_generation.py:1100-1395`

**Interfaces:**

- Consumes: `_native_mlx_vlm_load_kwargs`, `_native_mlx_vlm_template_kwargs`, `_native_mlx_vlm_generate_kwargs`, `_build_native_mlx_vlm_python_script`, `_diagnostics_model_blocks`, and `DiagnosticsPartitions`.
- Produces: `_build_parameterized_mlx_vlm_python_script(prompt_file_name: str, image_ref: str, run_args: argparse.Namespace | None) -> str`; one narrowly typed shared-context block builder.

- [ ] **Step 1: Add failing single-copy reproduction tests**

Generate diagnostics for one crash, two observed completions, one indeterminate
attempt, and one clean completion. Use an exact multiline prompt. Assert:

```python
assert diagnostics.count(prompt) == 1
assert diagnostics.count("from mlx_vlm.generate import generate") == 1
assert diagnostics.count("Canonical parameterised Python reproduction") == 1
assert "--prompt-file prompt.txt" in diagnostics
assert "python -m mlx_vlm.generate" not in diagnostics
```

Also assert every highlighted model ID and full resolved revision appears in the
shared model/revision table. Retain the existing per-crash issue-draft assertions
that require a direct CLI command and direct single-model Python script.

- [ ] **Step 2: Run the reproduction tests and verify failure**

Run:

```bash
pytest src/tests/test_report_generation.py -k "reproduction or crash_diagnostics" -q
```

Expected: aggregated diagnostics contain one CLI command and Python script per
highlighted result, so the single-copy assertions fail.

- [ ] **Step 3: Build one parameterised single-model script**

Implement a generated script that uses `argparse` with:

```text
model                 positional model ID
--revision            optional resolved revision
--image               required image path
--prompt-file         required UTF-8 prompt file
```

The script must read the prompt with `Path.read_text(encoding="utf-8")`, call
the existing mlx-vlm `load`, `apply_chat_template`, and `generate` sequence, and
embed the common load/template/generate kwargs obtained from the existing
helpers. Apply `args.revision` to a copied `LOAD_KWARGS` only when supplied. It
must run exactly one model per invocation so it does not create sequential Metal
state interactions.

- [ ] **Step 4: Add the shared report context blocks**

Implement `_diagnostics_shared_context_blocks(*, prompt: str, highlighted_results: Sequence[PerformanceResult], model_provenance: Mapping[str, ModelProvenanceRecord], library_versions: LibraryVersionDict, system_info: Mapping[str, str], image_path: Path | None, run_args: argparse.Namespace | None) -> tuple[ReportBlock, ...]` to return:

- `ReportSection("Prompt", (ReportCodeBlock(prompt),), ...)` with prose telling
  the reader to save it as `prompt.txt`;
- a highlighted model/revision `ReportTable`;
- one `ReportSection` containing the parameterised script;
- one components/system `ReportTable` built with
  `_collect_report_component_rows`.

Append these blocks once after all evidence and clean context. Do not include
the prompt or a reproduction block inside `_diagnostics_model_blocks`.

Update `_diagnostics_evidence_blocks` so aggregate entries no longer append
direct reproduction and narrow its final signature to
`_diagnostics_evidence_blocks(report_context: HtmlReportContext, *,`
`run_args: argparse.Namespace | None) -> tuple[ReportBlock, ...]`.
`generate_diagnostics_report` renders evidence followed by the shared context.
The HTML document renders the same evidence in its
maintainer section and the same shared context inside the existing provenance
section, replacing the old prompt/component-only HTML assembly rather than
duplicating it.

- [ ] **Step 5: Preserve complete direct crash issue drafts**

Build each individual crash draft from the shared crash evidence blocks, then
append its existing direct CLI and `_build_native_mlx_vlm_python_script` output
plus provenance. This keeps one-model issue drafts copy/paste runnable while the
aggregated `diagnostics.md` stays compact.

Delete the obsolete aggregated per-model reproduction wrappers and any now-dead
Markdown-only provenance assembly. Remove imports, helpers, or suppressions that
become unused.

- [ ] **Step 6: Run the complete report-generation suite**

Run:

```bash
pytest src/tests/test_report_generation.py src/tests/test_markdown_formatting.py src/tests/test_html_formatting.py -q
```

Expected: all report tests pass; highlighted output and crash evidence remain
complete; the aggregate prompt/script appear once; direct crash drafts retain
their direct reproduction.

- [ ] **Step 7: Commit shared reproduction context**

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_markdown_formatting.py src/tests/test_html_formatting.py
git commit -m "refactor: share diagnostics reproduction context"
```

---

### Task 5: Document roles and verify the complete change

**Files:**

- Modify: `src/README.md`
- Modify: `docs/IMPLEMENTATION_GUIDE.md`
- Modify: `.github/copilot-instructions.md`
- Modify: `CHANGELOG.md`
- Verify only: `src/output/`

**Interfaces:**

- Consumes: the final gallery and diagnostics contracts from Tasks 1-4.
- Produces: maintainer/user guidance and a fully verified, net-shorter implementation.

- [ ] **Step 1: Update report-role documentation**

Document these exact roles in the existing output/report sections:

- `model_gallery.md`: chooser plus readable and exact complete output for model
  comparison;
- `diagnostics.md`: self-contained mlx-vlm issue report with expanded crashes,
  collapsed highlighted evidence, clean-run context, and shared reproduction;
- `results.jsonl` and `check_models.log`: exhaustive machine and operational
  evidence.

In `.github/copilot-instructions.md`, add the maintenance rule that issue-ready
diagnostics must not repeat run-wide prompt/reproduction context per model and
must use existing report blocks rather than parallel Markdown/HTML builders.

- [ ] **Step 2: Update the changelog**

Under `[Unreleased]`, record both the user-visible formatting/diagnostic change
and the shared-renderer simplification/type tightening. Do not bump the release
version unless separately requested.

- [ ] **Step 3: Format and clear local lint findings**

Run:

```bash
make format
make -C src lint-fix
make lint
git diff --check
```

Expected: Ruff and whitespace checks pass without new suppressions.

- [ ] **Step 4: Run the full quality gate**

Run:

```bash
make quality
```

Expected: Ruff, mypy, ty, pyrefly, Vulture, suppression audit, Skylos audit,
pytest, ShellCheck, and Markdown lint all pass.

- [ ] **Step 5: Run explicit Skylos advisory checks**

Run:

```bash
cd src && skylos . -a --llm
cd .. && make skylos-danger-llm
```

Expected: both commands report no findings and no new ignore markers,
threshold changes, or suppressions are present.

- [ ] **Step 6: Verify simplification and artefact hygiene**

Run:

```bash
wc -l src/check_models.py
git status --short src/output
git diff --check
git diff --stat
```

Expected:

- `src/check_models.py` is fewer than 16,214 lines;
- `src/output/` has no validation-induced changes;
- no whitespace errors exist;
- the deleted format-specific diagnostic/reproduction code outweighs the new
  shared primitives and builders.

- [ ] **Step 7: Commit documentation and final cleanup**

```bash
git add .github/copilot-instructions.md CHANGELOG.md docs/IMPLEMENTATION_GUIDE.md src/README.md src/check_models.py src/tests
git commit -m "docs: describe issue-ready report roles"
```

- [ ] **Step 8: Run post-commit verification**

Run `make quality`, `cd src && skylos . -a --llm`, and
`make skylos-danger-llm` again on the exact committed tree. Confirm a clean
working tree and report the final line-count delta. Do not regenerate the model
matrix as part of this deterministic implementation plan.
