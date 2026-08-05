# Diagnostics Usefulness Recommendations

**Status:** Recommendations (not an approved implementation plan)  
**Date:** 2026-07-31  
**Branch context:** Written against the issue-ready reporting surfaces on
`codex/issue-ready-reports` / retained `0.8.9` artifact contract.

## Purpose

Capture product and report-shape recommendations for making check_models more
useful to two audiences:

1. **mlx-vlm maintainers** — people who paste diagnostics into upstream issues.
2. **Model selectors** — people choosing a local VLM for image description /
   catalog metadata on Apple Silicon.

This note does **not** change runtime behavior by itself. Implement from a
focused design/plan before coding.

For day-to-day agent workflows that already match upstream mlx-vlm operator
habits (minimal native CLI repros, server `curl` isolation, issue templates, and
the server-supported HF cache filter), see:

- `.agents/skills/native-mlx-vlm-repro/SKILL.md`
- `.agents/skills/upstream-mlx-vlm-issues/SKILL.md`
- `.agents/skills/hf-cache-mlx-vlm-models/SKILL.md`

Those skills use this repo’s **conda + pip** conventions, not upstream `uv run`
examples.

## Current strengths (do not regress)

The retained surfaces already provide the hard parts:

| Strength | Where |
| -------- | ----- |
| One immutable assessment shared by all artifacts | `execution` / `usability` / `maintainer_status` |
| Exact evidence preservation | JSONL schema `2.0`, gallery raw fences, crash drafts |
| Crash issue drafts with native CLI + Python repro | `src/output/issues/issue_*.md` |
| Skim-first aggregate diagnostics | `reports/diagnostics.md` |
| Facts-only gallery chooser + readable/raw output | `reports/model_gallery.md`, `reports/results.html` |
| Mechanical observation codes with exact `details` | JSONL + quality detectors |
| Explicit non-goals | No semantic caption scores, no winners/grades |

Primary reader guidance already in [src/README.md](../../src/README.md):

- Use **gallery / HTML** to choose models and compare complete output.
- Use **diagnostics** as a skim-first mlx-vlm issue body.
- Use **JSONL + run.json** as machine contracts.

## Audience A — mlx-vlm maintainers

### Gaps observed on real runs

- A full matrix (for example 62 models, 1 crash, ~36 observation cases) produces
  a large `diagnostics.md`. Maintainers paste issues; they do not want a novel.
- Issue drafts are created only for hard crashes (`execution == "crashed"`).
  Severe completed failures (empty/repeated output) stay in the aggregate only.
- Crash tracebacks center on local `check_models.py` frames. Useful for harness
  debugging; noisy for upstream.
- Observation-heavy runs list models individually instead of clustering by
  failure signature.
- `index.md` does not link `issues/`.
- `results.history.jsonl` is append-only and unused by current reports (no
  regression signal).

### Recommendations

#### 1. Split “file this” vs “browse this” (P0)

Add a GitHub-sized paste body alongside the full workbook:

| Artifact | Contents |
| -------- | -------- |
| `diagnostics-issue.md` (or equivalent mode) | Summary + triage + **expanded crashes only** + **one** shared repro + env/provenance. Hard size budget suitable for GitHub. |
| `diagnostics.md` (current) | Full maintainer workbook, including collapsed observation evidence |
| Optional `observation_clusters.md` | Group by observation signature, not only by model |

Keep complete evidence available; change the **default paste target**.

#### 2. Cluster observations by signature (P1)

Before the per-model list, emit themes such as:

```markdown
## Observation clusters (this run)

### repeated_output + token_cap_truncation (n=12)
Signature: hits max_tokens with dominating repeated fragment
Example models: …
Exact fragment (first hit): "…"
Minimal repro: model X @ rev … + shared script
```

Turn dozens of collapsed entries into a handful of bug themes.

#### 3. Strip harness-local noise from paste bodies (P0)

For paste-ready drafts and slim issue bodies:

- Prefer **root exception + first frames inside `mlx_vlm` / transformers /
  model code**.
- Collapse the full harness stack under “Harness stack (optional)”.
- Keep phase, package, root error, and resolved revision above the fold.

Example punch line that should lead:

> `Loaded processor has no image_processor; expected multimodal processor`
> @ `processor_load` / rev `…`

#### 4. Factual upstream-vs-harness bucket (P1)

Not blame scoring—derive a factual bucket from existing phase/package tags:

| Bucket | When |
| ------ | ---- |
| `likely_model_or_package` | Missing `image_processor`, bad config, load failure in model package |
| `likely_runtime_mlx_vlm` | Failure in generate / apply_chat_template / mlx-vlm runtime path |
| `harness_preflight` | check_models validator raised before native generate |
| `environment` | Metal OOM, disk, permissions |
| `indeterminate_connectivity` | Already retained, not filed as model crash |

Emit in drafts and triage so maintainers know whether to open mlx-vlm, a model
card, or ignore.

#### 5. Optional severe completed-failure drafts (P2, gated)

Keep default = crashes only. Optional flag, for example:

```text
--issue-drafts=crashes|severe|none
```

Where `severe` may also draft:

- `empty_output`
- `repeated_output` with token cap
- optionally pure `prompt_instruction_echo` with almost no real content

Still **no** automatic drafts for every `missing_requested_sections` case
(often prompt/contract friction, not a crash).

#### 6. One-click paste packing (P3)

- Suggested GitHub title + labels in front matter (model id, mlx-vlm version).
- Compact versions block: `mlx`, `mlx-vlm`, `mlx-lm`, macOS, chip, peak memory.
- Example `gh issue create --body-file …` using the slim body.
- Link issue drafts from `index.md`.
- Sanitize home paths to `~` consistently in paste surfaces.

#### 7. “Known good neighbor” on crashes (P2)

For each crash, one factual line:

> Same run: N usable, M observed-unusable; example related success: `model`
> (usable, T s, P GB).

Answers “is the whole stack broken?” without extra prose sections.

#### 8. Cross-run regression signal from history (P2)

Even a minimal diff against `results.history.jsonl` helps:

- New crash vs last run on same machine / image / prompt hash.
- Observation first seen after component revision X.

A short diagnostics subsection or `regression.md` beats asking humans to diff
JSONL.

## Audience B — model selectors (image description)

### Gaps observed on real runs

- Many rows marked `unusable` still produce good plain descriptions. They fail
  the **strict Title/Description/Keywords catalog contract**, not “can describe
  the image.”
- Chooser sort is avoid-first: a long unusable wall sits above usable models.
- No side-by-side field comparison; humans scroll long vertical evidence.
- Single image + single prompt overfits; history is not used for selection.
- Resource tables exist (lowest memory / fastest), but there is no one-screen
  Pareto shortlist or family/quant rollup.

### Recommendations

#### 1. Separate catalog compliance from description presence (P1)

Keep current `usability` for the strict catalog lane. Add parallel **mechanical**
axes (still non-semantic), for example:

| Axis | Meaning |
| ---- | ------- |
| `structure_ok` | Required sections present / no fatal contract breaks (current usable*) |
| `description_present` | Non-empty descriptive prose (length + not pure instruction echo + not pure repetition) |
| `wrapper_noise` | Thinking / special / role tokens present |
| `resource_class` | Peak GB buckets (for example ≤8 / ≤24 / ≤48 / >48) |

Gallery defaults can shortlist by description presence + resources, with catalog
compliance as a filter—not the only grade.

#### 2. First-class plain-description eval lane (P1)

Existing lanes: `triage` / `blind` / `assisted`. Add or advertise:

- `describe` — short plain caption, no Title/Keywords scaffolding  
  **or** promote `--eval-mode triage` as the “pick a captioner” path with its
  own gallery heading.

Optional dual retained views:

- **Catalog gallery** (current)
- **Description gallery** (plain caption prompt, ranked by completeness + resources)

#### 3. Usable-first chooser + Pareto shortlist (P0)

Default selector order:

1. Recommended shortlist (usable; top N by mem × latency × tokens)
2. Usable with caveats
3. Interesting failures (real prose preview but structural fail)
4. Full avoid list collapsed

One-screen Pareto examples (facts only):

```text
Best low-memory usable:   …
Best balanced usable:     …
Fastest usable:           …
```

Human still judges caption quality; the tool only shortlists by structure,
resources, and preview facts.

#### 4. Side-by-side compare for N models (P1)

CLI (`--compare a,b,c`) and/or HTML multi-select:

| Field | Model A | Model B | Model C |
| ----- | ------- | ------- | ------- |
| Title / Description / Keywords | … | … | … |
| Peak GB / e2e / TPS | … | … | … |
| Observations | … | … | … |

Parse catalog fields when present; otherwise show full text once.

#### 5. Family / quant rollup (P2)

Compact matrix so users shop architectures, not only HF ids:

| Family | Variants usable | Best mem | Best TPS | Notes |
| ------ | --------------- | -------- | -------- | ----- |

Answers “is this architecture bad, or this quant/build?”

#### 6. Constraint-aware navigational blurb (P2)

Template filled from run facts (not ML ranking):

> On this image/prompt: **N usable / M unusable / K crash**.  
> If you need **&lt;8 GB**: consider …  
> If you need **strict Title/Description/Keywords**: filter `usable` only.  
> If you need **any decent caption**: also inspect structural-fail rows whose
> preview is real prose.

#### 7. Multi-image / multi-prompt smoke (P2)

Even a tiny matrix beats single-image overfitting:

- Three built-in fixtures: simple object, text-in-image, busy scene; **or**
- `--images dir/` with a per-image usability matrix.

#### 8. Shortlist export pack (P3)

- `shortlist.md` — top usable + previews + re-run commands for just those models
- `shortlist.json` — ids, revisions, mem, tps, usability, output
- Stable HTML deep links (for example `?usability=usable&max_gb=16`)

## Cross-cutting recommendations

1. **Prompt-contract appendix in diagnostics** — exact required labels, max
   tokens, eval lane, and what triggers `missing_requested_sections`. Prevents
   “model is broken” issues that are really strict catalog friction.
2. **Observation → human one-liners** — keep machine codes in JSONL; show plain
   language in Markdown/HTML
   (`repeated_output` → “Generation stuck in a loop until max tokens.”).
3. **Rebuild reports from JSONL + run.json** — documented regeneration without
   re-running Metal; speeds report UX iteration and sharing.
4. **Artifact size honesty** — footer with bytes/lines for slim vs full vs
   gallery so people paste the right file.
5. **Do not reintroduce scores/winners** — prefer filters, clusters, shortlists,
   and side-by-side text (aligned with the 0.8.9 removal of grade/scorecard
   pipelines).

## Suggested priority

| Priority | Change | Primary audience | Why |
| -------- | ------ | ---------------- | --- |
| P0 | Slim paste-ready issue body + root-error-first crash drafts | Maintainers | Actually gets pasted into GitHub |
| P0 | Gallery usable-first shortlist + Pareto resource picks | Selectors | Cuts long-row fatigue |
| P1 | Observation clustering by signature | Maintainers | Dozens of rows → few themes |
| P1 | Side-by-side compare for N models | Selectors | Real caption choice |
| P1 | Plain-description lane and/or dual usability axes | Selectors | Fixes “unusable but good caption” |
| P2 | Family/quant matrix | Selectors | Matches how people shop |
| P2 | History regression subsection | Maintainers | “New since last mlx-vlm” |
| P2 | Multi-image matrix | Both | Less single-image overfitting |
| P2 | Upstream-vs-harness bucket; known-good neighbor | Maintainers | Faster triage |
| P3 | `gh issue create` helper + index links to drafts | Maintainers | Friction removal |
| P3 | JSONL→report rebuild; shortlist export | Both | Faster iteration and sharing |

## Explicit non-goals (still)

- Semantic caption scoring / LLM-as-judge as a retained grade
- Auto-filing GitHub issues without human review
- Drafting issues for every structural observation by default
- Restoring global winner scorecards or capability score pipelines
- Truncating complete highlighted crash/observation evidence solely to hit a
  fixed byte target in the **full** workbook (size budgets belong on the slim
  paste body)

## Implementation notes (when scheduled)

- Stay inside the intentional `src/check_models.py` monolith and existing report
  block / assessment pipeline.
- Reuse cached `ResultAssessment`; do not reclassify in renderers.
- Prefer presentation and packing changes before new detectors.
- Add focused tests in existing `src/tests/test_report_generation.py`,
  `test_markdown_formatting.py`, `test_html_formatting.py`, and related files.
- Route validation outputs to temp or gitignored `test_*` paths; do not rewrite
  tracked `src/output/` assets just to prove a change.
- Update [src/README.md](../../src/README.md) decision semantics and
  [CHANGELOG.md](../../CHANGELOG.md) when behavior ships.
- Follow [.github/copilot-instructions.md](../../.github/copilot-instructions.md)
  quality gate order before any real-model acceptance matrix.

## Related docs

- [Issue-ready reporting design](archive/superpowers/specs/2026-07-26-issue-ready-reporting-design.md) (archived)
- [Implementation guide](../IMPLEMENTATION_GUIDE.md)
- [CLI / output contract](../../src/README.md)
- [Contributing](../CONTRIBUTING.md)
