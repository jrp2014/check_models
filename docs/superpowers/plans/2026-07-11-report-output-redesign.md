# Report Output and Assisted Evaluation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an issue-ready mlx-vlm diagnostic report and defensible local
VLM selection reports by distinguishing metadata provenance, visual input
burden, task outcome, and ranking policy while retaining `results.html`.

**Architecture:** Extend the existing `ReportRenderContext`, diagnostics
snapshot/issue-cluster pipeline, prompt diagnostics, metadata agreement, and
repro-command builders. Introduce only shared views that replace repeated
derivation, then make Markdown, HTML, TSV, JSONL, history, and issue artifacts
consume those views. Delete superseded render/scoring helpers in the same task
that introduces their canonical replacement.

**Tech Stack:** Python 3.13, mlx-vlm/MLX runtime types, Pillow image metadata,
dataclasses and `TypedDict`, pytest, Ruff, mypy, ty, Pyrefly, Skylos,
markdownlint-cli2.

**Design:**
[`2026-07-11-report-output-redesign-design.md`](../specs/2026-07-11-report-output-redesign-design.md)

## Global Constraints

- Activate the `mlx-vlm` Conda environment before every Python or Make command.
- Keep `src/check_models.py` as the intentional single-file monolith.
- Add tests only to existing `src/tests/test_*.py` modules.
- Keep JSONL format `2.0` and history format `1.0`; new fields are optional.
- Preserve all current filenames and CLI flags, including deprecated
  `stress`/`quality` input aliases.
- Keep `results.html` as a primary, standalone human-readable artifact.
- Do not add an external LLM judge or automatic GitHub issue submission.
- Use `quality_config.yaml`/`QualityThresholds` for thresholds and weights.
- Reuse `ReportRenderContext`, `DiagnosticsSnapshot`, `IssueCluster`, existing
  repro specs, report block renderers, history, and safe artifact writers.
- Do not create renderer-specific scoring or a parallel diagnostic pipeline.
- Each production task must delete or fold superseded logic before commit.
- Prefer a net reduction in `src/check_models.py`; document any unavoidable net
  production-code growth with added/deleted counts and rationale.
- Route test-generated output to `tmp_path` or gitignored `test_*` paths.
- Update `CHANGELOG.md` under `[Unreleased]`.

---

## File Map

- `src/check_models.py`: all production data contracts, prompt building,
  quality/scoring, diagnostics, recommendation views, renderers, serialization,
  and final orchestration.
- `src/check_models_data/quality_config.yaml`: assisted-score weights and prompt
  burden thresholds.
- `src/tests/test_pure_logic_functions.py`: assisted/blind prompt contracts and
  metadata provenance parsing.
- `src/tests/test_quality_analysis.py`: provenance-aware scores, labels,
  burden classification, and text-sanity evidence gating.
- `src/tests/test_process_image_mock.py`: exception-chain capture and prompt
  diagnostics populated during model execution.
- `src/tests/test_report_generation.py`: diagnostics, issue views, selection,
  gallery, HTML, and cross-artifact consistency.
- `src/tests/test_jsonl_output.py`: optional JSONL/history fields and backward
  compatibility.
- `src/tests/test_tsv_output.py`: new scalar columns and stable escaping.
- `src/README.md`: user-facing lane, scoring, burden, and report-role docs.
- `docs/IMPLEMENTATION_GUIDE.md`: canonical-view reuse and code-size policy.
- `docs/CONTRIBUTING.md`: report fixture and consistency-test guidance.
- `CHANGELOG.md`: maintainer-facing release note.
- `src/output/`: one intentional final benchmark snapshot refresh after all
  behavior and tests stabilize.

---

### Task 1: Build one metadata-provenance view and use it in prompts

**Files:**

- Modify: `src/check_models.py:4080-4405`
- Modify: `src/check_models.py:6480-6540`
- Modify: `src/check_models.py:22120-22280`
- Test: `src/tests/test_pure_logic_functions.py:339-440`
- Test: `src/tests/test_quality_analysis.py:313-455`

**Interfaces:**

- Produces: `MetadataProvenance`,
  `_build_metadata_provenance(metadata: MetadataDict | None) -> MetadataProvenance`.
- Preserves: `TrustedHintBundle` for prompt-derived compatibility checks.
- Consumed by: Tasks 2, 3, 6, and 7.

- [ ] **Step 1: Add failing provenance and prompt tests**

Add these tests to the existing prompt test class and quality-analysis module:

```python
def test_assisted_prompt_separates_authoritative_context_from_draft(
    mod: types.ModuleType,
) -> None:
    args = argparse.Namespace(prompt=None, eval_mode="assisted")
    prompt = mod.prepare_prompt(
        args,
        {
            "title": "Deben Estuary at Woodbridge",
            "description": "Two boats on a river.",
            "keywords": "Deben Estuary, Woodbridge, boats, river",
            "date": "2026-07-04",
            "time": "19:10:04",
            "gps": "52.0,-1.0",
        },
    )

    assert "Authoritative context:" in prompt
    assert "Location terms: Deben Estuary, Woodbridge" in prompt
    assert "Capture date/time: 2026-07-04 19:10:04" in prompt
    assert "Draft descriptive metadata:" in prompt
    assert "Existing description: Two boats on a river." in prompt
    assert "Treat this draft as fallible" in prompt


def test_metadata_provenance_keeps_location_out_of_draft_terms() -> None:
    provenance = check_models._build_metadata_provenance(
        {
            "title": "Deben Estuary at Woodbridge",
            "description": "Two sailing boats with trees behind.",
            "keywords": "Deben Estuary, Woodbridge, England, boats, trees",
        }
    )

    assert "Deben Estuary" in provenance.authoritative_terms
    assert "Woodbridge" in provenance.authoritative_terms
    assert "boats" in {term.casefold() for term in provenance.draft_terms}
    assert "woodbridge" not in {term.casefold() for term in provenance.draft_terms}
```

- [ ] **Step 2: Run the focused tests and confirm the red state**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_pure_logic_functions.py::TestPreparePrompt src/tests/test_quality_analysis.py -k "provenance or authoritative_context" -q'
```

Expected: failure because `MetadataProvenance` and the two prompt headings do
not exist.

- [ ] **Step 3: Add the shared provenance contract and builder**

Place the dataclass beside `TrustedHintBundle`. Use the existing term splitters
instead of adding another metadata tokenizer:

```python
@dataclass(frozen=True)
class MetadataProvenance:
    """Authoritative structured context plus fallible descriptive draft fields."""

    authoritative_terms: tuple[str, ...] = ()
    capture_date: str | None = None
    capture_time: str | None = None
    gps: str | None = None
    draft_title: str | None = None
    draft_description: str | None = None
    draft_keywords: tuple[str, ...] = ()
    draft_terms: tuple[str, ...] = ()

    @property
    def has_authoritative_context(self) -> bool:
        return bool(
            self.authoritative_terms or self.capture_date or self.capture_time or self.gps
        )

    @property
    def has_draft(self) -> bool:
        return bool(self.draft_title or self.draft_description or self.draft_keywords)


def _clean_metadata_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value).strip(" ,")
    return cleaned or None


def _build_metadata_provenance(metadata: MetadataDict | None) -> MetadataProvenance:
    if not metadata:
        return MetadataProvenance()

    title = _clean_metadata_text(metadata.get("title"))
    description = _clean_metadata_text(metadata.get("description"))
    keyword_text = _clean_metadata_text(metadata.get("keywords")) or ""
    keyword_terms = _split_catalog_keywords(keyword_text)
    draft_keywords, authoritative_keywords = _partition_hint_terms(keyword_terms)

    authoritative_terms = list(authoritative_keywords)
    draft_terms = list(draft_keywords)
    for free_text in (title, description):
        if free_text is None:
            continue
        descriptive, contextual = _partition_hint_terms(_extract_hint_signal_terms(free_text))
        draft_terms.extend(descriptive)
        authoritative_terms.extend(contextual)

    return MetadataProvenance(
        authoritative_terms=tuple(_dedupe_preserve_order(authoritative_terms)),
        capture_date=_clean_metadata_text(metadata.get("date")),
        capture_time=_clean_metadata_text(metadata.get("time")),
        gps=_clean_metadata_text(metadata.get("gps")),
        draft_title=title,
        draft_description=description,
        draft_keywords=tuple(draft_keywords),
        draft_terms=tuple(_dedupe_preserve_order(draft_terms)),
    )
```

If the existing configured location-term matching does not recognize the test
locations, extend the existing `nonvisual_location_terms` list in
`quality_config.yaml`; do not add a second location dictionary in Python.

- [ ] **Step 4: Replace duplicated prompt/scoring field selection**

Change `_build_cataloguing_prompt()` to call `_build_metadata_provenance()` once.
Render `Authoritative context:` and `Draft descriptive metadata:` only when the
respective data exists. Keep the `Context:` marker on the first context heading
so existing parsing remains available during migration:

```python
provenance = _build_metadata_provenance(metadata if include_metadata_hints else None)
if provenance.has_authoritative_context:
    parts.extend(["", "Context: Authoritative context:"])
    if provenance.authoritative_terms:
        parts.append("- Location terms: " + ", ".join(provenance.authoritative_terms))
    capture_value = " ".join(
        value for value in (provenance.capture_date, provenance.capture_time) if value
    )
    if capture_value:
        parts.append(f"- Capture date/time: {capture_value}")
    if provenance.gps:
        parts.append(f"- GPS: {provenance.gps}")
    parts.append(
        "- Use this factual context where it improves the catalogue record; do not "
        "claim that contextual facts are visually observable."
    )

if provenance.has_draft:
    parts.extend(["", "Draft descriptive metadata:"])
    if provenance.draft_title:
        parts.append(f"- Existing title: {provenance.draft_title}")
    if provenance.draft_description:
        parts.append(f"- Existing description: {provenance.draft_description}")
    if provenance.draft_keywords:
        parts.append("- Existing keywords: " + ", ".join(provenance.draft_keywords))
    parts.append(
        "- Treat this draft as fallible. Retain supported details, correct errors, "
        "and add important visible information."
    )
```

Refactor `_build_metadata_scoring_inputs()` to consume the same provenance view
and delete its duplicate title/description/keyword/date/GPS extraction loop.

- [ ] **Step 5: Run prompt and quality tests**

Run the same focused command from Step 2.

Expected: all selected tests pass; existing blind tests continue to prove that
neither provenance class is exposed.

- [ ] **Step 6: Commit the provenance/prompt slice**

```bash
git add src/check_models.py src/check_models_data/quality_config.yaml src/tests/test_pure_logic_functions.py src/tests/test_quality_analysis.py
git commit -m "feat: distinguish metadata provenance in assisted prompts"
```

---

### Task 2: Replace blanket metadata borrowing with enrichment components

**Files:**

- Modify: `src/check_models.py:1160-1230`
- Modify: `src/check_models.py:4580-4770`
- Modify: `src/check_models.py:6480-6760`
- Modify: `src/check_models.py:6820-7445`
- Modify: `src/check_models_data/quality_config.yaml:100-150`
- Test: `src/tests/test_quality_analysis.py:313-455`
- Test: `src/tests/test_quality_analysis.py:1070-1320`
- Test: `src/tests/test_report_generation.py:1920-1990`

**Interfaces:**

- Consumes: `MetadataProvenance` from Task 1.
- Produces: optional `context_integration_score`, `draft_improvement_score`, and
  `visual_description_score` on `MetadataAgreementMetrics` and JSONL payloads.
- Preserves: existing metadata-agreement fields for backward compatibility.

- [ ] **Step 1: Add failing assisted-enrichment tests**

```python
def test_assisted_location_use_is_context_integration_not_borrowing() -> None:
    metadata = {
        "title": "Deben Estuary at Woodbridge",
        "description": "Two boats on a river.",
        "keywords": "Deben Estuary, Woodbridge, boats, river",
    }
    text = (
        "Title: Sailboats on Deben Estuary at Woodbridge\n"
        "Description: Two moored sailboats stand before a wooded bank.\n"
        "Keywords: sailboats, Deben Estuary, Woodbridge, river, wooded bank"
    )

    metrics = compute_metadata_agreement(text, metadata)

    assert metrics.context_integration_score is not None
    assert metrics.context_integration_score > 0
    assert metrics.nonvisual_penalty == 0


def test_verbatim_draft_is_low_improvement_not_hallucination() -> None:
    metadata = {
        "description": "Two boats on a river.",
        "keywords": "boats, river",
    }
    text = (
        "Title: Two boats on a river\n"
        "Description: Two boats on a river.\n"
        "Keywords: boats, river"
    )
    metrics = compute_metadata_agreement(text, metadata)
    analysis = analyze_generation_text(
        text,
        generated_tokens=20,
        prompt_tokens=120,
        prompt="Return Title, Description, and Keywords.",
    )

    assert analysis.metadata_borrowing is False
    assert metrics.draft_improvement_score is not None
    assert metrics.draft_improvement_score < 50
    assert not analysis.hallucination_issues
```

- [ ] **Step 2: Run the selected tests and verify they fail**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_quality_analysis.py src/tests/test_report_generation.py -k "context_integration or low_improvement or assisted_location" -q'
```

Expected: failure because the new component fields and provenance-aware labels
are absent.

- [ ] **Step 3: Add optional score fields and centralized weights**

Add validated YAML/default fields:

```yaml
  assisted_weight_visual_description: 0.35
  assisted_weight_context_integration: 0.25
  assisted_weight_draft_improvement: 0.20
  assisted_weight_output_quality: 0.20
  low_draft_improvement_score: 40.0
```

Extend `QualityThresholds.__post_init__()` to require each weight in `[0, 1]`
and their sum within `FLOAT_ZERO_EPSILON` of `1.0`.

Extend `MetadataAgreementMetrics` with optional fields rather than changing
existing field meanings:

```python
context_integration_score: float | None = None
draft_improvement_score: float | None = None
visual_description_score: float | None = None
assisted_enrichment_score: float | None = None
```

When `_populate_result_quality_analysis()` already refreshes a result with
metadata, copy only the presentation fields needed by existing review helpers
onto `GenerationQualityAnalysis`; do not make prompt-only analysis parse the
metadata blocks a second time.

- [ ] **Step 4: Implement the component calculations from existing metrics**

Add these shared helpers beside metadata scoring:

```python
def _score_authoritative_context(
    text: str,
    provenance: MetadataProvenance,
) -> tuple[float | None, tuple[str, ...]]:
    terms = tuple(_dedupe_preserve_order(provenance.authoritative_terms))
    if not terms:
        return None, ()
    normalized = _normalize_phrase_for_matching(text)
    matched = tuple(term for term in terms if _context_term_present(term, normalized))
    target = min(len(terms), QUALITY.metadata_alignment_partial_match_cap)
    return round(min(len(matched) / max(target, 1), 1.0) * 100.0, 1), matched


def _score_draft_improvement(
    text: str,
    provenance: MetadataProvenance,
) -> float | None:
    draft_parts = [
        value
        for value in (provenance.draft_title, provenance.draft_description)
        if value
    ]
    if provenance.draft_keywords:
        draft_parts.append(", ".join(provenance.draft_keywords))
    if not draft_parts:
        return None
    baseline = compute_cataloging_utility("\n".join(draft_parts), None)
    generated = compute_cataloging_utility(text, None)
    delta = float(generated["utility_score"]) - float(baseline["utility_score"])
    return round(max(0.0, min(100.0, 50.0 + delta)), 1)


def _weighted_available_score(
    components: Sequence[tuple[float | None, float]],
) -> float | None:
    available = [(value, weight) for value, weight in components if value is not None]
    if not available:
        return None
    weight_total = sum(weight for _value, weight in available)
    return round(sum(value * weight for value, weight in available) / weight_total, 1)
```

Use the existing description/caption utility score as the visual-description
heuristic and current hygiene/contract score as output quality. Normalize
available weights rather than converting unavailable components to zero.

- [ ] **Step 5: Replace assisted-mode presentation labels**

Keep the old `metadata_borrowing` JSON field readable, but stop setting it for
correct authoritative-context use. Set it only for the existing misleading
date/GPS-as-visible case, and render `unverified-context-copy` for that case.
Render `low-draft-improvement` when the new score is below the configured
threshold. Delete branches that separately add “nonvisual metadata reused” to
gallery, review, selection, and diagnostics; derive those surfaces from the
canonical review evidence list.

- [ ] **Step 6: Prevent structured coordinates from becoming issue-grade loops**

Add a normalizer used by `_text_sanity_numeric_loop_issue()`:

```python
_STRUCTURED_NUMERIC_METADATA_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b|"
    r"\b\d{1,3}(?:\.\d+)?°\s*[NSEW]\b|\b\d+\s*/\s*\d+\b)",
    re.IGNORECASE,
)


def _remove_structured_numeric_metadata(text: str) -> str:
    return _STRUCTURED_NUMERIC_METADATA_RE.sub(" ", text)
```

Call it before numeric-token counting. Add exact tests for decimal coordinates,
degrees/minutes/seconds coordinates, timestamps, dimensions, and exposure
fractions, plus one true repeated-number loop that must still fire.

- [ ] **Step 7: Run focused quality/report tests**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_quality_analysis.py src/tests/test_report_generation.py -k "metadata or context or draft or numeric or sanity" -q'
```

Expected: all selected tests pass and existing blind-mode metadata-leak tests
remain green.

- [ ] **Step 8: Commit the scoring/detector slice**

```bash
git add src/check_models.py src/check_models_data/quality_config.yaml src/tests/test_quality_analysis.py src/tests/test_report_generation.py
git commit -m "fix: score assisted metadata enrichment by provenance"
```

---

### Task 3: Add one canonical prompt/input burden view

**Files:**

- Modify: `src/check_models.py:1160-1230`
- Modify: `src/check_models.py:1980-2115`
- Modify: `src/check_models.py:7280-7445`
- Modify: `src/check_models.py:8450-8480`
- Modify: `src/check_models.py:11040-11840`
- Modify: `src/check_models.py:19220-19295`
- Modify: `src/check_models_data/quality_config.yaml:120-145`
- Test: `src/tests/test_quality_analysis.py:407-455`
- Test: `src/tests/test_process_image_mock.py:484-610`
- Test: `src/tests/test_report_generation.py:2500-2560`
- Test: `src/tests/test_jsonl_output.py:625-680`

**Interfaces:**

- Produces: `PromptBurden`,
  `_prompt_burden_for_result(result, image_profile) -> PromptBurden`.
- Extends: `PromptDiagnostics`, `ImageInputProfile`, JSONL review/diagnostics.
- Consumed by: Tasks 5, 6, and 7.

- [ ] **Step 1: Add failing burden-classification tests**

```python
def test_large_image_short_text_is_visual_input_burden() -> None:
    analysis = check_models.GenerationQualityAnalysis(
        is_repetitive=False,
        repeated_token=None,
        hallucination_issues=[],
        is_verbose=False,
        formatting_issues=[],
        has_excessive_bullets=False,
        bullet_count=0,
        is_context_ignored=False,
        missing_context_terms=[],
        is_refusal=False,
        refusal_type=None,
        is_generic=False,
        specificity_score=1.0,
        has_language_mixing=False,
        language_mixing_issues=[],
        has_degeneration=False,
        degeneration_type=None,
        has_fabrication=False,
        fabrication_issues=[],
        prompt_tokens_total=16700,
        prompt_tokens_text_est=430,
        prompt_tokens_nontext_est=16270,
    )
    result = check_models.PerformanceResult(
        model_name="org/visual-heavy",
        generation=None,
        success=True,
        quality_analysis=analysis,
        prompt_diagnostics=check_models.PromptDiagnostics(image_placeholder_count=1),
    )
    profile = check_models.ImageInputProfile(width=9504, height=6336, megapixels=60.2)

    burden = check_models._prompt_burden_for_result(result, profile)

    assert burden.kind == "visual_input"
    assert burden.text_tokens_est == 430
    assert burden.nontext_tokens_est == 16270
    assert burden.visual_tokens_est is None
    assert burden.source == "estimated_nontext"


def test_small_image_long_text_is_text_burden() -> None:
    analysis = check_models.GenerationQualityAnalysis(
        is_repetitive=False,
        repeated_token=None,
        hallucination_issues=[],
        is_verbose=False,
        formatting_issues=[],
        has_excessive_bullets=False,
        bullet_count=0,
        is_context_ignored=False,
        missing_context_terms=[],
        is_refusal=False,
        refusal_type=None,
        is_generic=False,
        specificity_score=1.0,
        has_language_mixing=False,
        language_mixing_issues=[],
        has_degeneration=False,
        degeneration_type=None,
        has_fabrication=False,
        fabrication_issues=[],
        prompt_tokens_total=4200,
        prompt_tokens_text_est=3900,
        prompt_tokens_nontext_est=300,
    )
    result = check_models.PerformanceResult(
        model_name="org/text-heavy",
        generation=None,
        success=True,
        quality_analysis=analysis,
        prompt_diagnostics=check_models.PromptDiagnostics(image_placeholder_count=1),
    )
    profile = check_models.ImageInputProfile(width=640, height=480, megapixels=0.3)

    burden = check_models._prompt_burden_for_result(result, profile)

    assert burden.kind == "text"
    assert burden.nontext_ratio is not None
    assert burden.nontext_ratio < 0.5
```

Construct these records directly in the existing test module; do not add a
production test fixture.

- [ ] **Step 2: Run focused tests and verify the red state**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_quality_analysis.py src/tests/test_process_image_mock.py src/tests/test_jsonl_output.py -k "burden or prompt_diagnostics" -q'
```

Expected: failure because `PromptBurden` and its classifier do not exist.

- [ ] **Step 3: Move burden thresholds into configuration**

Add and validate:

```yaml
  heavy_nontext_prompt_ratio: 0.50
  mixed_prompt_burden_ratio_floor: 0.25
```

Delete `HEAVY_NON_TEXT_PROMPT_RATIO_THRESHOLD` after all call sites use
`QUALITY.heavy_nontext_prompt_ratio`. Continue using the existing
`long_prompt_tokens_threshold` for the minimum substantial burden.

- [ ] **Step 4: Add the canonical burden view**

```python
type PromptBurdenKind = Literal["normal", "visual_input", "text", "mixed", "unknown"]


@dataclass(frozen=True)
class PromptBurden:
    kind: PromptBurdenKind
    total_tokens: int | None = None
    text_tokens_est: int | None = None
    nontext_tokens_est: int | None = None
    visual_tokens_est: int | None = None
    template_tokens_est: int | None = None
    nontext_ratio: float | None = None
    source: str = "unavailable"
    original_width: int | None = None
    original_height: int | None = None
    processed_width: int | None = None
    processed_height: int | None = None
    patch_count: int | None = None


def _prompt_burden_for_result(
    result: PerformanceResult,
    image_profile: ImageInputProfile | None,
) -> PromptBurden:
    analysis = result.quality_analysis
    total = analysis.prompt_tokens_total if analysis is not None else None
    text_est = analysis.prompt_tokens_text_est if analysis is not None else None
    nontext_est = analysis.prompt_tokens_nontext_est if analysis is not None else None
    ratio = nontext_est / total if total and nontext_est is not None else None
    diagnostics = result.prompt_diagnostics
    placeholders = diagnostics.image_placeholder_count if diagnostics is not None else 0
    substantial = total is not None and total >= QUALITY.long_prompt_tokens_threshold

    kind: PromptBurdenKind = "unknown" if total is None else "normal"
    if substantial and ratio is not None:
        if placeholders and ratio >= QUALITY.heavy_nontext_prompt_ratio:
            kind = "visual_input"
        elif text_est is not None and text_est >= QUALITY.long_prompt_tokens_threshold:
            kind = "text" if ratio < QUALITY.mixed_prompt_burden_ratio_floor else "mixed"
        else:
            kind = "mixed"

    return PromptBurden(
        kind=kind,
        total_tokens=total,
        text_tokens_est=text_est,
        nontext_tokens_est=nontext_est,
        nontext_ratio=ratio,
        source="estimated_nontext" if nontext_est is not None else "unavailable",
        original_width=image_profile.width if image_profile is not None else None,
        original_height=image_profile.height if image_profile is not None else None,
        processed_width=(diagnostics.processed_image_width if diagnostics is not None else None),
        processed_height=(diagnostics.processed_image_height if diagnostics is not None else None),
        patch_count=diagnostics.image_patch_count if diagnostics is not None else None,
    )
```

Do not populate `visual_tokens_est` by subtraction. Leave it `None` until an
upstream processor exposes a separable count.

- [ ] **Step 5: Extend existing prompt/image diagnostics without a second pass**

Add optional processed-dimension/patch fields to `PromptDiagnostics`. Populate
explicit `--resize-shape` dimensions in `_build_prompt_diagnostics()` and leave
dynamic processor dimensions unavailable. Extend `ImageInputProfile` only with
optional processed fields that are actually measured; do not reopen the image
per model.

Replace human strings “long prompt” and “nontext burden” in review/diagnostic
helpers with labels from `_prompt_burden_for_result()`. Delete duplicated ratio
classification branches in `_review_owner_specific_next_action()`,
`_review_focus_parts()`, and long-context symptom formatting.

- [ ] **Step 6: Serialize the burden view additively**

Add optional scalar fields to JSONL review/prompt diagnostics and TSV:

```python
prompt_burden_kind: str
prompt_burden_source: str
processed_image_width: int | None
processed_image_height: int | None
image_patch_count: int | None
```

Keep old `prompt_tokens_nontext_est` and `nontext_prompt_ratio` fields readable.

- [ ] **Step 7: Run focused tests**

Run the Step 2 command plus:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_tsv_output.py -k "burden or prompt or image" -q'
```

Expected: burden classification, unavailable-field, serialization, and existing
prompt tests pass.

- [ ] **Step 8: Commit the burden slice**

```bash
git add src/check_models.py src/check_models_data/quality_config.yaml src/tests/test_quality_analysis.py src/tests/test_process_image_mock.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py
git commit -m "feat: distinguish visual and textual prompt burden"
```

---

### Task 4: Preserve exception chronology and derive one failure narrative

**Files:**

- Modify: `src/check_models.py:1270-1320`
- Modify: `src/check_models.py:1980-2075`
- Modify: `src/check_models.py:19570-19615`
- Modify: `src/check_models.py:20655-20725`
- Modify: `src/check_models.py:24920-25000`
- Test: `src/tests/test_process_image_mock.py:350-470`
- Test: `src/tests/test_jsonl_output.py:1280-1335`
- Test: `src/tests/test_report_generation.py:3020-3335`

**Interfaces:**

- Reuses: `_exception_chain()`, `_root_cause_exception()`,
  `_maintainer_triage_for_result()`.
- Produces: `FailureException`, `FailureNarrative`,
  `_build_failure_narrative(result: PerformanceResult) -> FailureNarrative`.
- Consumed by: Tasks 5, 6, and 7.

- [ ] **Step 1: Add failing chained-exception tests**

```python
def test_build_failure_result_preserves_exception_chain_order() -> None:
    try:
        try:
            raise IndexError("token id 999 outside detokenizer table")
        except IndexError:
            raise RuntimeError("METAL command buffer out of memory")
    except RuntimeError as error:
        wrapped = ValueError("generation failed")
        wrapped.__cause__ = error
        result = check_models._build_failure_result(
            model_name="org/model",
            error=wrapped,
            captured_output=None,
        )

    assert [entry.exception_type for entry in result.exception_chain] == [
        "IndexError",
        "RuntimeError",
        "ValueError",
    ]
    narrative = check_models._build_failure_narrative(result)
    assert narrative.task_outcome == "crashed"
    assert narrative.primary_exception == "IndexError: token id 999 outside detokenizer table"
    assert narrative.secondary_exceptions == (
        "RuntimeError: METAL command buffer out of memory",
        "ValueError: generation failed",
    )
```

- [ ] **Step 2: Run focused failure tests and verify the red state**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_process_image_mock.py src/tests/test_jsonl_output.py src/tests/test_report_generation.py -k "exception_chain or failure_narrative or upstream_traceback" -q'
```

Expected: failure because exception-chain records and the narrative builder are
absent.

- [ ] **Step 3: Materialize the already-traversed chain once**

```python
@dataclass(frozen=True)
class FailureException:
    exception_type: str
    module: str
    message: str


@dataclass(frozen=True)
class FailureNarrative:
    task_outcome: Literal["crashed"]
    phase: str | None
    stage: str | None
    primary_exception: str
    secondary_exceptions: tuple[str, ...]
    suspected_owner: str
    owner_confidence: MaintainerConfidence


def _failure_exception_record(error: BaseException) -> FailureException:
    return FailureException(
        exception_type=type(error).__name__,
        module=type(error).__module__,
        message=str(error),
    )
```

Add `exception_chain: tuple[FailureException, ...] = ()` to `PerformanceResult`.
In `_build_failure_result()`, existing `_exception_chain(error)` yields outer to
inner, so store `tuple(reversed(...))` once. Derive legacy `root_error_*` fields
from the first stored entry and retain all existing fields.

- [ ] **Step 4: Build the canonical narrative from existing triage**

```python
def _format_failure_exception(entry: FailureException) -> str:
    return f"{entry.exception_type}: {entry.message}"


def _build_failure_narrative(result: PerformanceResult) -> FailureNarrative:
    chain = result.exception_chain
    if chain:
        primary = _format_failure_exception(chain[0])
        secondary = tuple(_format_failure_exception(entry) for entry in chain[1:])
    else:
        primary = f"{result.root_error_type or result.error_type or 'Error'}: " \
            f"{result.root_error_message or result.error_message or 'Unknown failure'}"
        secondary = ()
    triage = _maintainer_triage_for_result(result)
    chain_owners = {
        owner
        for entry in chain
        if (owner := _attribute_error_to_package(entry.message)) != "unknown"
    }
    suspected_owner = triage["suspected_owner"] if triage is not None else "unknown"
    owner_confidence = triage["confidence"] if triage is not None else "low"
    if len(chain_owners) > 1:
        suspected_owner = "unresolved: " + "/".join(sorted(chain_owners))
        owner_confidence = "low"
    elif len(chain) > 1 and owner_confidence == "high":
        owner_confidence = "medium"
    return FailureNarrative(
        task_outcome="crashed",
        phase=result.failure_phase,
        stage=result.error_stage,
        primary_exception=primary,
        secondary_exceptions=secondary,
        suspected_owner=suspected_owner,
        owner_confidence=owner_confidence,
    )
```

Delete report helpers that independently choose between root and wrapper error
text; make them consume this narrative.

Extend the existing MLX attribution patterns with the concrete Metal/OOM phrases
already emitted by this project (`insufficient memory`, `out of memory`, and
`kIOGPUCommandBufferCallbackErrorOutOfMemory`) so a mixed mlx-vlm/MLX chain is
reported as unresolved rather than confidently assigned to one package.

- [ ] **Step 5: Serialize the chain additively**

Add optional `exception_chain` to the JSONL result and repro failure payload:

```python
record["exception_chain"] = [
    {
        "type": entry.exception_type,
        "module": entry.module,
        "message": entry.message,
    }
    for entry in result.exception_chain
]
```

Omit the field when the chain is empty. Keep root/wrapper fields unchanged for
old readers.

- [ ] **Step 6: Run focused tests**

Run the Step 2 command.

Expected: chain order, owner confidence, legacy fields, and failure formatting
all pass.

- [ ] **Step 7: Commit the failure slice**

```bash
git add src/check_models.py src/tests/test_process_image_mock.py src/tests/test_jsonl_output.py src/tests/test_report_generation.py
git commit -m "fix: preserve failure exception chronology"
```

---

### Task 5: Rebuild diagnostics from existing issue/repro primitives

**Files:**

- Modify: `src/check_models.py:12620-12760`
- Modify: `src/check_models.py:13310-15240`
- Modify: `src/check_models.py:15620-16380`
- Modify: `src/check_models.py:25740-26655`
- Test: `src/tests/test_report_generation.py:2700-3335`
- Test: `src/tests/test_report_generation.py:3930-4250`
- Test: `src/tests/test_report_generation.py:5240-5820`

**Interfaces:**

- Consumes: `PromptBurden` from Task 3 and `FailureNarrative` from Task 4.
- Reuses: `DiagnosticsSnapshot`, `IssueCluster`, `_build_issue_clusters()`,
  `_render_issue_queue_table()`, `build_native_mlx_vlm_repro_command_spec()`,
  `_issue_image_hash_line()`, and issue acceptance helpers.
- Produces: a compact issue-ready `diagnostics.md`; no new output file.

- [ ] **Step 1: Add failing issue-ready diagnostics tests**

```python
def test_diagnostics_is_self_contained_mlx_vlm_issue(tmp_path: Path) -> None:
    failure = _make_failure_with_details(
        "org/crashing-model",
        error_msg="generation failed",
        error_stage="Model Error",
        error_package="mlx-vlm",
        traceback_str="Traceback\nIndexError: token id out of range",
    )
    output = tmp_path / "diagnostics.md"

    written = generate_diagnostics_report(
        results=[failure],
        filename=output,
        versions={"mlx-vlm": "0.6.5", "mlx": "0.32.1"},
        system_info={"GPU/Chip": "Apple M5 Max", "RAM": "128 GB"},
        prompt="Describe the image.",
        image_path=tmp_path / "fixture.jpg",
        run_args=Namespace(max_tokens=500, temperature=0.0, top_p=1.0),
    )

    content = output.read_text(encoding="utf-8")
    assert written is True
    assert "## Crash / Failure Matrix" in content
    assert "Task outcome: crashed" in content
    assert "python -m mlx_vlm.generate" in content
    assert "## Expected and Actual Behaviour" in content
    assert "## Models Not Flagged" not in content
    assert "Total model runtime (sum)" not in content


def test_model_only_text_quality_is_not_confirmed_mlx_vlm_issue(tmp_path: Path) -> None:
    result = _make_quality_success("org/model-only", with_quality_issue=False)
    analysis = cast("GenerationQualityAnalysis", result.quality_analysis)
    result = replace(
        result,
        quality_analysis=replace(
            analysis,
            text_sanity_issue_type="numeric_loop",
            owner="model",
            user_bucket="avoid",
        ),
    )
    output = tmp_path / "diagnostics.md"

    generate_diagnostics_report(
        results=[result],
        filename=output,
        versions={"mlx-vlm": "0.6.5"},
        system_info={},
        prompt="Describe the image.",
    )

    content = output.read_text(encoding="utf-8")
    assert "Confirmed mlx-vlm issues" not in content
    assert "Model/config observations" in content
```

- [ ] **Step 2: Run diagnostics tests and verify the red state**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py -k "self_contained_mlx_vlm or model_only_text_quality or crash_issue or issue_queue or reproducibility" -q'
```

Expected: failures because diagnostics still points to issue drafts for native
repros and includes internal coverage/unflagged sections.

- [ ] **Step 3: Add shared cluster confidence and filing scope helpers**

Avoid a new issue hierarchy. Derive views from existing clusters:

```python
def _issue_cluster_confidence(cluster: IssueCluster) -> MaintainerConfidence:
    confidences = [
        triage["confidence"]
        for result in cluster.results
        if (triage := _maintainer_triage_for_result(result)) is not None
    ]
    if "high" in confidences:
        return "high"
    if "medium" in confidences:
        return "medium"
    return "low"


def _issue_cluster_is_mlx_vlm_scope(cluster: IssueCluster) -> bool:
    return cluster.owner in {"mlx-vlm", "mlx"}
```

Use confidence labels `confirmed`, `probable`, and `untriaged` only for routing;
never change the task outcome of failed results.

- [ ] **Step 4: Extract one compact native repro block from issue-draft code**

Refactor `_issue_repro_section()` so both issue drafts and diagnostics call:

```python
def _native_repro_block(
    cluster: IssueCluster,
    *,
    prompt: str,
    image_path: Path | None,
    run_args: argparse.Namespace | None,
) -> ReportSection:
    representative = _issue_cluster_representative(cluster)
    model_name = representative.model_name if representative is not None else "unknown/model"
    image_ref = _issue_repro_image_ref(image_path=image_path, run_args=run_args)
    command = build_native_mlx_vlm_repro_command_spec(
        model_name=model_name,
        prompt=prompt,
        image_ref=image_ref,
        run_args=run_args,
    )
    return ReportSection(
        title="Native mlx-vlm reproduction",
        blocks=(ReportCodeBlock(command.shell_command(), language="bash"),),
    )
```

Keep the full Python/config appendix only in individual issue drafts. Delete
duplicate native command assembly from diagnostics.

- [ ] **Step 5: Replace diagnostics section orchestration**

Rework `generate_diagnostics_report()` to assemble, in order:

1. impact/run conditions;
2. all failures using `_build_failure_narrative()`;
3. compact mlx-vlm-scope issue matrix;
4. per-cluster expected/actual/native repro/acceptance blocks;
5. relevant `_prompt_burden_for_result()` evidence;
6. probable/model-config appendix; and
7. compact environment.

Delete calls to `_diagnostics_unflagged_models_section()` and the general runtime
coverage section from this report. If those helpers have no remaining callers,
delete them. Retain runtime summaries in `results.md`/HTML where they are useful.

- [ ] **Step 6: Tighten issue-cluster admission for text sanity**

Change `_collect_text_sanity_results()` or `_build_issue_clusters()` so an
issue-grade text-sanity cluster requires both a detector type and a supporting
excerpt predicate. Reuse detector output; do not re-run regexes in the renderer.
Coordinates excluded in Task 2 must remain ordinary model-quality evidence.

- [ ] **Step 7: Run the diagnostics and issue-report suites**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py -k "diagnostic or issue or repro or failure" -q'
```

Expected: all diagnostic, native repro, issue draft, safe-link, and failure
evidence tests pass.

- [ ] **Step 8: Commit the diagnostics consolidation**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "refactor: make diagnostics an issue-ready report"
```

---

### Task 6: Build one current-run recommendation view

**Files:**

- Modify: `src/check_models.py:12760-12930`
- Modify: `src/check_models.py:16900-18390`
- Modify: `src/check_models.py:22820-23190`
- Test: `src/tests/test_report_generation.py:432-710`
- Test: `src/tests/test_report_generation.py:1635-1990`
- Test: `src/tests/test_report_generation.py:2370-2560`

**Interfaces:**

- Consumes: provenance scores from Task 2, burden view from Task 3, and failure
  narrative from Task 4.
- Produces: `ModelRecommendationView`,
  `_build_model_recommendation_views(context) -> tuple[ModelRecommendationView, ...]`.
- Replaces: `ModelSelectionRow` and duplicated current-run ranking derivation.
- Consumed by: Task 7 HTML/Markdown/JSON alignment.

- [ ] **Step 1: Add failing eligibility and policy tests**

```python
def test_recommendation_view_excludes_crash_from_usable_policies() -> None:
    failed = _make_failure("org/crashed")
    passed = _make_success("org/passed")
    context = _build_report_render_context(
        results=[failed, passed],
        prompt="Describe the image.",
        eval_mode="blind",
    )

    views = check_models._build_model_recommendation_views(context)
    by_model = {view.result.model_name: view for view in views}

    assert by_model["org/crashed"].compatibility == "crashed"
    assert by_model["org/crashed"].eligible is False
    assert by_model["org/passed"].eligible is True


def test_model_selection_names_each_ranking_policy(tmp_path: Path) -> None:
    result = _make_metadata_agreement_result()
    context = _build_report_render_context(
        results=[result],
        prompt="Create title, description, and keywords.",
        metadata={"description": "Two boats", "keywords": "boats, river"},
        eval_mode="assisted",
    )
    output = tmp_path / "model_selection.md"

    generate_model_selection_report(
        [result],
        output,
        prompt="Create title, description, and keywords.",
        report_context=context,
    )

    content = output.read_text(encoding="utf-8")
    assert "Policy: reliability-gated assisted enrichment" in content
    assert "Evidence scope: 1 image, 1 current run" in content
```

- [ ] **Step 2: Run focused selection tests and verify the red state**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py -k "recommendation_view or ranking_policy or model_selection or scorecard" -q'
```

Expected: failure because the canonical view and policy labels are absent.

- [ ] **Step 3: Define the compact canonical current-run view**

```python
type CompatibilityStatus = Literal["crashed", "integration-warning", "clean"]


@dataclass(frozen=True)
class ModelRecommendationView:
    result: PerformanceResult
    compatibility: CompatibilityStatus
    eligible: bool
    eligibility_reason: str
    visual_score: float | None
    context_score: float | None
    draft_improvement_score: float | None
    output_score: float | None
    assisted_enrichment_score: float | None
    first_token_latency_s: float | None
    total_time_s: float | None
    generation_tps: float | None
    peak_memory_gb: float | None
    burden: PromptBurden
    caveats: tuple[str, ...]
```

Build it from existing review, metadata agreement, utility rows, runtime
diagnostics, and burden helpers. Eligibility is false for failures, harness
issues, `avoid`, unusable/empty output, and scores below existing configured
chooser thresholds. Delete `ModelSelectionRow` after migrating its callers.

- [ ] **Step 4: Implement named policy selectors**

Use functions returning existing views, not additional row classes:

```python
def _eligible_recommendations(
    views: Sequence[ModelRecommendationView],
) -> list[ModelRecommendationView]:
    return [view for view in views if view.eligible]


def _rank_reliability_gated_enrichment(
    views: Sequence[ModelRecommendationView],
) -> list[ModelRecommendationView]:
    return sorted(
        _eligible_recommendations(views),
        key=lambda view: (
            view.assisted_enrichment_score
            if view.assisted_enrichment_score is not None
            else -1.0,
            -(view.total_time_s if view.total_time_s is not None else math.inf),
            view.result.model_name,
        ),
        reverse=True,
    )


def _rank_under_memory_budget(
    views: Sequence[ModelRecommendationView],
    budget_gb: float,
) -> list[ModelRecommendationView]:
    return [
        view
        for view in _rank_reliability_gated_enrichment(views)
        if view.peak_memory_gb is not None and view.peak_memory_gb <= budget_gb
    ]
```

Add a Pareto helper that removes a view only when another eligible view is at
least as good in enrichment score, total time, and peak memory and strictly
better in one. Keep weights/thresholds in configuration.

- [ ] **Step 5: Group comparable repository variants**

Add one conservative family-key helper for the comparison section only:

```python
_MODEL_VARIANT_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"-(?:bf16|fp16|[348]-?bit|6bit|8bit|mxfp4|mxfp8|nvfp4|qat-4bit|qat-8bit)$",
    re.IGNORECASE,
)


def _model_family_key(model_name: str) -> str:
    owner, separator, model = model_name.partition("/")
    normalized = _MODEL_VARIANT_SUFFIX_RE.sub("", model)
    return f"{owner}/{normalized}" if separator else normalized
```

Only render a family comparison when two or more current results share a key.
Do not merge their scores or histories.

- [ ] **Step 6: Rebuild selection/capability surfaces from the view**

Change model selection quick choosers and current-run capability signals to
consume `ModelRecommendationView`. Each section prints its policy and evidence
scope. Keep historical aggregation in `ModelCapabilityRunSignal`; only replace
its duplicated current-run derivation.

Delete superseded selection score/caveat helpers with no remaining callers.

- [ ] **Step 7: Run selection/capability tests**

Run the Step 2 command.

Expected: crash gating, policy labels, memory budgets, Pareto ordering, family
comparisons, and lane-filtered capability history pass.

- [ ] **Step 8: Commit the recommendation consolidation**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "refactor: unify current-run model recommendations"
```

---

### Task 7: Align HTML, Markdown, JSONL, TSV, history, and artifact roles

**Files:**

- Modify: `src/check_models.py:10400-11980`
- Modify: `src/check_models.py:16640-16920`
- Modify: `src/check_models.py:17400-18400`
- Modify: `src/check_models.py:24700-25260`
- Modify: `src/check_models.py:27000-27720`
- Test: `src/tests/test_report_generation.py:950-1135`
- Test: `src/tests/test_report_generation.py:4430-5065`
- Test: `src/tests/test_jsonl_output.py:90-420`
- Test: `src/tests/test_tsv_output.py:31-240`

**Interfaces:**

- Consumes: all canonical views from Tasks 1-6.
- Preserves: current artifact paths, JSONL/history versions, safe escaping, and
  standalone HTML behavior.
- Produces: consistent primary and secondary artifacts.

- [ ] **Step 1: Add failing cross-artifact consistency tests**

```python
def test_html_and_markdown_share_task_outcome_and_policy(tmp_path: Path) -> None:
    results = [_make_failure("org/crashed"), _make_success("org/passed")]
    context = _build_report_render_context(
        results=results,
        prompt="Describe the image.",
        eval_mode="blind",
    )
    html_path = tmp_path / "results.html"
    selection_path = tmp_path / "model_selection.md"

    generate_html_report(
        results,
        html_path,
        versions={},
        prompt="Describe the image.",
        total_runtime_seconds=1.0,
        report_context=context,
    )
    generate_model_selection_report(
        results,
        selection_path,
        prompt="Describe the image.",
        report_context=context,
    )

    html_text = html_path.read_text(encoding="utf-8")
    selection_text = selection_path.read_text(encoding="utf-8")
    assert "org/crashed" in html_text
    assert "crashed" in html_text
    assert "reliability-gated" in html_text
    assert "reliability-gated" in selection_text
```

Add JSONL/TSV assertions for the same compatibility status, component scores,
burden kind, and owner confidence.

- [ ] **Step 2: Run cross-artifact tests and verify the red state**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py -k "share_task_outcome or compatibility_status or enrichment or burden or html" -q'
```

Expected: failure because current renderers derive several facts independently.

- [ ] **Step 3: Extend `ReportRenderContext` with reusable views**

Add cached tuples with empty defaults:

```python
recommendations: tuple[ModelRecommendationView, ...] = ()
diagnostics_snapshot: DiagnosticsSnapshot = dataclass_field(
    default_factory=DiagnosticsSnapshot
)
issue_clusters: tuple[IssueCluster, ...] = ()
```

Populate them once in `_build_report_render_context()`: build one
`DiagnosticsSnapshot`, derive `issue_clusters` from that instance, and reuse
both objects in finalization and every renderer. Change the diagnostics writer
to accept the cached snapshot/context rather than calling the classifiers
again. Do not store renderer-ready escaped strings.

- [ ] **Step 4: Make HTML consume canonical blocks and views**

Use `render_report_html()` for shared `ReportSection`/`ReportTable` blocks where
Markdown already uses `render_report_markdown()`. Keep the existing complete
HTML results table, image preview, filtering, escaping, and standalone CSS.
Replace only independently recomputed crash, run-contract, burden, and winner
facts.

Delete duplicate HTML-only selection/failure calculations after their callers
use `ReportRenderContext.recommendations` and `_build_failure_narrative()`.

- [ ] **Step 5: Add optional machine fields without schema bumps**

Extend existing `TypedDict` records and serializers with:

```python
compatibility_status: str
context_integration_score: float | None
draft_improvement_score: float | None
visual_description_score: float | None
assisted_enrichment_score: float | None
prompt_burden_kind: str
prompt_burden_source: str
owner_confidence: str
```

Omit optional values when unavailable. Keep JSONL `2.0`, history `1.0`, old
metadata agreement fields, and old prompt ratio fields.

- [ ] **Step 6: Clarify artifact roles without deleting compatibility outputs**

Update `generate_output_index_report()` so the primary routes are diagnostics,
HTML, model selection, gallery, and JSONL. Label results Markdown, review,
capability, TSV, history, bundles, and issue drafts as supporting artifacts.
Ensure secondary reports consume canonical views and cannot name a contradictory
winner without a different explicit policy label.

- [ ] **Step 7: Run the full report/serialization suite**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py src/tests/test_html_formatting.py -q'
```

Expected: all report, escaping, schema, history, HTML, TSV, and output-role tests
pass.

- [ ] **Step 8: Commit the artifact-alignment slice**

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py src/tests/test_html_formatting.py
git commit -m "refactor: align human and machine report views"
```

---

### Task 8: Document, refresh snapshots, audit code size, and verify

**Files:**

- Modify: `src/README.md`
- Modify: `docs/IMPLEMENTATION_GUIDE.md`
- Modify: `docs/CONTRIBUTING.md`
- Modify: `CHANGELOG.md`
- Modify intentionally: `src/output/README.md`, `src/output/index.md`,
  `src/output/reports/diagnostics.md`, `src/output/reports/results.html`,
  `src/output/reports/results.md`, `src/output/reports/model_selection.md`,
  `src/output/reports/model_gallery.md`, `src/output/reports/review.md`,
  `src/output/reports/model_capabilities.md`, `src/output/results.jsonl`,
  `src/output/model_capabilities.json`, `src/output/run.json`, and
  `src/output/reports/results.tsv` only when regenerated by the accepted
  benchmark snapshot command.
- Test: existing full repository quality gate.

**Interfaces:**

- Consumes: completed implementation from Tasks 1-7.
- Produces: documented behavior, representative tracked output, code-size audit,
  and verified branch.

- [ ] **Step 1: Update user and contributor documentation**

Document these exact distinctions:

- authoritative location/capture context versus LLM-written draft metadata;
- blind versus assisted scoring semantics;
- visual-input, text, mixed, normal, and unavailable burden labels;
- crash outcome versus suspected owner/confidence;
- `diagnostics.md` as issue-ready mlx-vlm output;
- `results.html` as the retained complete HTML report;
- primary versus supporting artifacts; and
- single-image shortlist versus multi-run capability language.

Add the canonical-view/code-size rule to `IMPLEMENTATION_GUIDE.md` and report
fixture guidance to `CONTRIBUTING.md`.

- [ ] **Step 2: Add the changelog entry**

Under `[Unreleased]`, describe provenance-aware enrichment, burden attribution,
exception chronology, issue-ready diagnostics, recommendation consolidation,
and retained HTML output. Mention compatibility of existing schemas/paths.

- [ ] **Step 3: Run formatting and safe lint repair in required order**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make format'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make -C src lint-fix'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make lint'
```

Expected: Ruff format and lint pass with no new suppressions.

- [ ] **Step 4: Run focused suites once more before snapshot refresh**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_pure_logic_functions.py src/tests/test_quality_analysis.py src/tests/test_process_image_mock.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py src/tests/test_html_formatting.py -q'
```

Expected: all selected tests pass. Do not run them again after a successful full
quality gate.

- [ ] **Step 5: Refresh tracked benchmark artifacts intentionally**

Run the existing representative image benchmark with default production output
paths, only after focused tests are green:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && cd src && python -m check_models --image /Users/jrp/Pictures/Processed/20260704-181004_DSC00862_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --prefill-step-size 4096 --timeout 300.0 --verbose'
```

If the representative source image no longer exists, stop this step and ask
the maintainer for its replacement; do not silently substitute a different
benchmark input. Review the diff to confirm:

- diagnostics contains inline native repros and no unflagged-model inventory;
- the gemma-style chained failure reads chronologically;
- coordinate metadata is not called token soup;
- assisted rankings reward location integration and draft improvement;
- high non-text/image input is not called a lengthy textual prompt;
- HTML remains complete and internally consistent; and
- no private absolute path appears in copy/paste native commands.

- [ ] **Step 6: Perform the production-code size audit**

```bash
git diff --numstat origin/main...HEAD -- src/check_models.py
git diff --stat origin/main...HEAD -- src/check_models.py
```

Expected: production additions are offset by deletions from duplicate report,
scoring, issue, and recommendation logic. If `src/check_models.py` grows overall,
add a short “Code-size audit” paragraph to `CHANGELOG.md` stating the exact
added/deleted counts, which new upstream facts required growth, and which
duplicate helpers were removed.

- [ ] **Step 7: Run the complete quality gate**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make quality'
```

Expected: workflow validation, dependency sync, Ruff, mypy, suppression audit,
ty, Pyrefly, Vulture, Skylos, full pytest, shellcheck, and markdownlint all pass.

- [ ] **Step 8: Review repository state and commit the final integration**

```bash
git status --short
git diff --check
git add CHANGELOG.md src/README.md docs/IMPLEMENTATION_GUIDE.md docs/CONTRIBUTING.md src/output
git commit -m "docs: publish redesigned benchmark outputs"
```

Expected: only intentional documentation/snapshot changes enter this final
commit; no test-generated temporary files or unrelated user changes are staged.

- [ ] **Step 9: Final implementation handoff**

Report:

- focused and full quality results;
- production Python added/deleted counts;
- deleted or consolidated helper families;
- primary artifact paths;
- schema compatibility status; and
- any upstream measurements that remained unavailable and therefore render as
  estimates or `null`.
