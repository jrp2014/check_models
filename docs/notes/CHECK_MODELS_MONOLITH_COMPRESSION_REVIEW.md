# check_models.py Monolith Compression Review

**Date:** 2026-07-31  
**Target:** [`src/check_models.py`](../../src/check_models.py)  
**Scope:** Human review for compression, robustness, and tool-based diagnostic
analysis. The monolith structure is intentional; splitting into packages is out
of scope.

## Snapshot

| Metric | Value |
| ------ | ----- |
| File size | ~16.6k lines |
| Top-level functions | ~504 |
| Classes | ~103 |
| Exact AST body-duplicate groups | None meaningful |

Intentional monolith; static tools miss most redundancy because the real waste
is **near-duplicate structure**, not identical clones.

## Section weight (where the mass lives)

| Section | ~Lines | Share |
| ------- | -----: | ----- |
| Result enrichment / history / finalization | 2,373 | largest |
| Model processing | 2,169 | |
| Console / system / EXIF-IPTC-XMP | 2,089 | |
| Diagnostics / report context | 1,858 | |
| CLI run helpers / logging | 1,814 | |
| App constants & core result types | 1,214 | |
| Metrics / scoring / field formatting | 1,268 | |
| Formatting / escaping / detectors | 962 | |
| Types / protocols / JSONL | 734 | |
| Imports / config | 732 | |
| Argparse | 694 | |
| Report generators | 592 | |
| Timing / logging plumbing | 142 | |

## Why tools say “no duplication”

Static tools look for:

- exact/near-identical AST bodies
- unused symbols
- simple clone pairs

This file mostly has:

1. **Policy cloned across formats** (Markdown chooser vs HTML chooser) with
   different render APIs
2. **Table-shaped logic written as long `getattr` / `if` chains** (native CLI +
   Python repro)
3. **Thin type-narrowing ladders** (`_generation_*_metric`) that look “used” and
   non-identical
4. **Single-use builders that are large and named** (argparse groups, diagnostics
   partitions) — intentional under the implementation guide, invisible to clone
   detectors
5. **Embedded CSS/JS strings** (~3.7k + ~1.3k chars) that hide structure from
   code-similarity tools

AST scan: **no meaningful exact body-duplicate groups**. The issue is
architectural redundancy, not copy-paste twins.

## Highest-value compression (ranked)

### 1. High — Unify gallery *policy*, keep thin format skins

Estimated savings: about 250–400 lines.

Near-parallel:

- `_render_gallery_chooser` (~141 lines)
- `_html_gallery_chooser` (~181 lines)

Same structure:

- sort by usability + model
- main chooser table
- avoid list
- lowest-memory usable
- fastest valid generation + average

Also parallel complete-evidence paths:

- `_generate_model_gallery_section`
- `_html_complete_gallery`
- `_render_gallery_model` / `_html_gallery_model`

**Compress by:** one pure function that returns ordered partitions + row DTOs;
MD/HTML only format cells.

**Bonus robustness bug:** HTML complete gallery uses a *local* `usability_order`
(unusable first) while Markdown and the chooser use
`_gallery_usability_sort_key` (usable-first). Comment claims “same stable
order”; behavior does not. Unifying the policy fixes this.

### 2. High — Data-drive native repro kwargs / CLI flags

Estimated savings: about 150–250 lines.

Today:

- `_native_mlx_vlm_generate_kwargs` — long `getattr` table already half
  data-driven
- `_build_native_mlx_vlm_cli_tokens` (~130 lines) — mostly hand-rolled pairs of
  the same idea
- parallel Python script builders (~100 lines combined)

**Compress by:** one declarative table, e.g.

```text
(attr, default, cli_flag, kwargs_key, kind: scalar|seq|bool|json)
```

Generate CLI tokens and `generate()` kwargs from that table. Same source of
truth as argparse field names where possible.

Also: **~43× `getattr(run_args, …)`** — hurts type checkers and makes flag drift
easy. Prefer a typed `RunSettings` / `ReproSettings` frozen dataclass filled once
from `Namespace`.

### 3. Medium–high — Collapse generation-metric ladder

Estimated savings: about 40–70 lines, plus clearer analysis.

Around the generation metric helpers:

- `_generation_numeric_metric`
- `_generation_int_metric`
- `_generation_float_metric`
- `_generation_text_value`
- `_generation_nonnegative_int_metric`
- `_generation_nonnegative_float_metric`
- `_generation_optional_nonnegative_float_metric`
- `_object_model_load_active_memory_gb` (one-liner over the optional float helper)

These are real, but the tower is tool-hostile (`getattr` + many names).

**Option A (compress):** one function

```text
_gen_metric(obj, name, *, kind, default, nonnegative=False)
```

**Option B (analyzer-friendly, may grow slightly):** map known
`GenerationResult` fields through a Protocol/`TypedDict` once in
`_extract_generation_performance_data` and stop re-getattr’ing field-by-field at
call sites.

Keep the **bool rejection** and nonnegative rules — those are robustness, not
noise.

### 4. Medium — Inline / delete trivial single-expression helpers

**~30–80 lines** (readability win more than raw LOC)

Good inline candidates (simple bodies; low conceptual weight):

| Helper | Shape | Notes |
| ------ | ----- | ----- |
| `_append_report_markdown_block` | `parts.extend(...)` | only used by `render_report_markdown` |
| `_markdown_emphasis` | `f"*{text}*"` | several uses; fine as local or constant pattern |
| `_html_attr` | f-string attr | keep if used often, or fold into `_html_status_attrs` |
| `_gallery_metric` | thin `format_field_value` | many uses — **don’t** inline everywhere; maybe alias |
| `_gallery_fact` / `_gallery_observation_labels` | 3-liners | optional |
| `_is_str_object_mapping` | TypeGuard one-liner | keep for typing *or* fold into `_as_str_object_mapping` |
| `_default_report_mode_policy` | one call | inline at field default |
| `_canonical_file_path`, memory sample one-liners | 3L | only if call sites stay readable |

**Do not** mass-inline every “refs≤1” function. Hundreds of private helpers are
single-use; most are 40–230 line *steps* (argparse groups, generation, issue
drafts). Inlining those would *increase* complexity and fight the implementation
guide.

### 5. Medium — Extract / table-drive big embedded presentation blobs

**~80–150 lines in the .py file** (LOC moves, not always deleted)

- `_html_filter_controls` — large JS/HTML string
- CSS in `_build_full_html_document`

Options:

- keep in-module but as module-level `Final` constants (clears function bodies
  for analyzers), or
- package data files under `check_models_data/` (smaller .py, slightly more
  packaging surface)

Doesn’t change behavior; improves susceptibility of surrounding Python to
tool-based diagnostic analysis.

### 6. Medium — Markdown blockquote neutralization

`_neutralize_markdown_blockquote_prefix` (~48 lines of nested if/else)

**Compress by:** ordered list of `(predicate, transform)` rules. Same behavior,
flatter control flow, easier tests.

### 7. Medium — Metadata / XMP / EXIF stack

~2k lines in console/metadata section; many small codecs (`_xmp_*`, IPTC, GPS,
mojibake).

Compression is limited without losing fail-soft behavior. Realistic wins:

- share one “decode bytes → str with mojibake repair” path (already partly there)
- table of EXIF/IPTC/XMP key → field mappings instead of parallel extractors

Estimate **~50–120 lines**, higher regression risk — do only with existing
robustness tests.

### 8. Lower — Argparse surface

`_add_model_prompt_generation_arguments` (~230L) and the runtime/console group
(~162L) are single-use by design.

Could be a list of `ArgumentSpec` dataclasses → loop `add_argument`. Saves
repetition, can improve consistency with repro flag tables (item 2). Risk:
help-text / argparse edge cases. **~80–150 lines** if done carefully.

### 9. Lower — Dual report contexts

`ReportRenderContext` vs `CanonicalHtmlReportContext` / `HtmlReportContext`
union.

Not huge LOC, but every consumer branches. Prefer one context with optional
fields, or always build full context. Improves analysis more than line count.

## What *not* to compress

Aligned with [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md) philosophy:

- **Don’t split the monolith** into packages for “size.”
- **Don’t inline large single-use conceptual steps** (`_run_model_generation`,
  `_classify_error`, `_build_cataloguing_prompt`, diagnostics partition
  builders).
- **Don’t merge MD/HTML *escaping* rules** into one generic escaper —
  differences are security/lint contracts.
- **Don’t delete shared `ReportBlock` render dispatch** — already the right
  compression of report formats; per-type functions are fine.
- **Don’t chase Vulture “unused” on TypedDict keys / Protocol methods** — known
  false positives here.
- **Don’t remove quality detectors** to save lines; they’re product behavior.

## Robustness (independent of compression)

1. **Gallery sort inconsistency** (above) — fix when unifying
   chooser/complete-evidence policy.
2. **`getattr(run_args, attr, default)` sprawl** — silent defaults when flags
   rename; typed settings object fails loudly.
3. **Re-running detectors in assessment** — `_assessment_observations` sometimes
   recomputes minimal/repetitive paths if `quality_analysis` is partial; prefer
   one authoritative analysis on the result.
4. **`cast("float", …)` in throughput averages** — replace with a narrow helper
   that already validated `generation_tps is not None`.
5. **HTML chooser vs Markdown columns** — HTML exposes execution / maintainer /
   prefill columns Markdown doesn’t; fine product-wise, but document as
   intentional so “unify” doesn’t accidentally drop HTML-only columns.
6. **Issue draft / native CLI flag matrix** — table-driven generation reduces
   “Python kwargs has X, CLI forgot X” drift (already partially mitigated by
   comments).

## Analyzer / tool friendliness

| Pain | Why tools struggle | Mitigation |
| ---- | ------------------ | ---------- |
| Many `getattr(` calls | dynamic attributes | Typed `GenerationResult` mapping + `RunSettings` |
| Hundreds of top-level functions | call-graph noise, “unused” noise | fewer micro-helpers; keep section landmarks |
| Huge string literals | no structural signal | constants or data files |
| Parallel MD/HTML policy | not exact clones | shared pure policy layer |
| `Namespace` everywhere | weak types | dataclass at boundary |
| Dual contexts / unions | extra branches | one context type |

Improving analyzability often **slightly increases** types/tables while
**decreasing** branching helpers — net LOC can still fall.

## Realistic savings (safe, no split)

| Tier | Scope | Est. net lines |
| ---- | ----- | -------------: |
| A | Trivial wrapper cleanup + sort-key fix + metric ladder | **80–150** |
| B | + gallery policy unification + repro/CLI tables | **400–700** |
| C | + argparse specs + CSS/JS out of functions + metadata tables | **800–1,200** |

Beyond ~1.2k lines you’re into behavior risk or “move code to other files,”
which this repo explicitly rejects for the monolith.

**Sweet spot:** Tier **B** — biggest readability/robustness return without
fighting project philosophy.

## Suggested order of work

1. Fix complete-gallery sort to use `_gallery_usability_sort_key` (small,
   correctness).
2. Extract chooser/complete-evidence **policy** shared by MD/HTML.
3. Introduce `RunSettings` + declarative native-repro flag table (CLI + kwargs +
   scripts).
4. Collapse generation metric accessors *or* centralize extraction once.
5. Only then prune 3-line cosmetics (`_append_report_markdown_block`, etc.).
6. Leave EXIF/XMP and quality detectors alone unless tests are green and scoped.

## Bottom line

Tools aren’t wrong that there are few **clones**. The monolith’s bulk is:

- real domain surface (metadata, mlx-vlm drift, reports, repro), plus
- **duplicated presentation policy** (gallery),
- **duplicated flag plumbing** (argparse vs native CLI vs Python repro), and
- a **long tail of tiny named wrappers** from type-checker / complexity
  pressure.

Best compression is **not** “delete single-line helpers everywhere.” It’s **one
policy layer + one settings/flag table**, with a few ladders collapsed. That
also fixes a real ordering inconsistency and makes mypy/Skylos/Vulture more
useful.

## Related docs

- [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md) — readable-first / helper
  extraction rules
- [DIAGNOSTICS_USEFULNESS_RECOMMENDATIONS.md](DIAGNOSTICS_USEFULNESS_RECOMMENDATIONS.md)
  — product-facing report shape notes
