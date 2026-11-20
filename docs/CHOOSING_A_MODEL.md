# Choosing a Captioning & Keywording Model

This guide explains how to read `check_models` reports when choosing a local MLX
Vision-Language Model (VLM) for photograph captioning and keywording on Apple
Silicon.

`check_models` records **objective mechanical checks and measured resource facts**
only — it does not score factual accuracy. Always inspect the actual generated
titles, descriptions, and keywords beside the photograph before adopting a model.

## 1. Where to look first

1. **[`src/output/index.md`](../src/output/index.md)** — one-screen summary of the
   run's evaluation lane, input image, completion counts, and links to all
   artifacts.
2. **[`src/output/reports/results.html`](../src/output/reports/results.html)** —
   self-contained local report (download or open locally in a browser) with
   interactive sorting, filters, inline reference image, and full outputs.
3. **[`src/output/reports/model_gallery.md`](../src/output/reports/model_gallery.md)**
   — GitHub-browsable version of the chooser table, *Output at a Glance*, and
   per-model complete evidence.
4. **[`src/output/issues/run_summary.md`](../src/output/issues/run_summary.md)** —
   concise sweep table including keyword counts, hint-copy share, and cross-run
   diffs.

## 2. Narrow by your Mac's memory and latency budget

Start with the hardware constraints that rule models in or out immediately:

- **Peak GB vs. unified memory:** leave headroom for macOS, your photo editor,
  and Metal working-set limits. As a practical rule of thumb, keep `Peak GB`
  comfortably below ~60–70% of total RAM (for example, `<= 10 GB` on a 16 GB Mac,
  `<= 16 GB` on 24 GB, `<= 24 GB` on 36 GB).
- **Total time and Prefill/first s (not just Gen TPS):** on large photographs
  (e.g., 40–66 MP JPEGs), dynamic-tiling architectures can expand a single image
  to 12,000–16,000+ prompt tokens and spend 30–60 s in prefill before emitting
  the first token. Fixed-resolution models often finish end-to-end in 3–8 s.
  If you want faster runs from high-tile models, test `--resize-shape` or
  `--processor-kwargs` (see [src/README.md](../src/README.md)).
- **Reasoning / thinking models:** models that generate a `<think>…</think>`
  block produce many more generated tokens (`Gen tok` of 400–900 vs. 80–140 for
  direct models), which increases total latency even when `Gen TPS` looks high.
  A properly closed thinking block followed by a complete answer is normal
  behaviour; an unclosed block that exhausts `max_tokens` is flagged as
  `incomplete thinking block` / `cut off at token limit`.

## 3. Read the evaluation lane before judging content

Every report states the run's **Evaluation lane** at the top:

- **`blind`** — the model sees only the image plus camera-recorded capture
  date/time and GPS (if present). Use `blind` runs to see what a model actually
  recognises from pixels alone.
- **`assisted`** — the prompt also includes existing title, description, and
  keyword hints from the image's IPTC/XMP metadata. Use `assisted` runs to see
  how well a model cleans up, refines, and extends draft catalogue metadata.
- **`triage`** — short smoke-test prompt (`Describe this image briefly.`) for
  quick compatibility checks, not captioning quality.

In an `assisted` run, pay close attention to two objective counts:

- **Keywords (`N (M new)` / `K from hints`):** distinguishes models that add
  genuinely new visual terms from models that copy all supplied keyword hints
  verbatim (including non-visual tags such as agency names if present in the
  source file).
- **Hint text (`%`):** percentage of the description's words appearing inside
  4-word runs copied from the prompt's description hint. High overlap is not
  automatically wrong, but it means the description does not prove independent
  visual understanding.

## 4. Check mechanical status, then read the actual text

Mechanical checks filter out broken structure and runtime faults before you spend
time reading captions:

- **`no concerns detected`** — the run completed, returned labelled `Title`,
  `Description`, and `Keywords` fields (in the `metadata` profile), had no
  duplicate keywords, no leaked control tokens, and did not truncate or loop.
  **This is not an accuracy verdict** — inspect the final answer against the
  photograph for misidentified subjects, invented breeds/locations, or missed
  details.
- **`concerns detected`** — completed with minor structural observations such as
  `duplicate keywords` or `prompt hint repeated`.
- **`major concerns: answer format`** — generated text, but missed requested
  sections, emitted extra preamble/duplicate answers, or hit `max_tokens`.
- **`major concerns: generation`** — empty response, repetition loop, or
  thinking-only output with no final answer.

## 5. Compare across multiple photographs

A single photograph cannot show how a model handles portraits, clutter, small
scene text, document OCR, or difficult lighting. Use the fixed five-image public
set in [`src/tools/image_set.tsv`](../src/tools/image_set.tsv) (see *Image set*
in [src/README.md](../src/README.md)) to compare your shortlisted models across
varied scenes, and run both `--eval-mode blind` and `--eval-mode assisted` on
the same photo if your workflow relies on existing metadata hints.

## 6. Record your own accuracy notes

Once you have inspected a model's output against the image, you can attach a
short human note tied to that exact image, model revision, and output digest:

```bash
cd src
python -m tools.add_accuracy_note MODEL_ID \
  --reviewer your-name \
  --note "Accurate subject and lighting; missed the tablet in the background."
```

Notes are stored in `src/output/accuracy_notes.yaml` (never generated
automatically) and appear beside that model in the gallery, HTML report, and run
summary for as long as the image, model revision, and output remain unchanged.
