# Output Improvements for Bug Hunting and Quality Evaluation

Based on a review of the actual logs and reports (specifically `results.md` and `check_models.log`), the following improvements are proposed to make the tool more effective.

## 1. Bug Identification (Library & Environment)

The current logs capture errors but present them as raw stack traces. We can parse these to provide actionable "Root Cause" and "Fix Suggestions".

### A. Intelligent Error Parsing

* **Problem:** Errors like `[metal::malloc]` (OOM) or `ImportError` are grouped generically as "processing" or "ValueError".
* **Proposal:** Implement an error classifier that regex-matches common failure patterns:
  * **OOM:** `[metal::malloc] ... greater than the maximum allowed` -> **Status:** `OOM` -> **Suggestion:** "Try quantization (4bit/8bit) or a smaller batch size."
  * **Missing Dependency:** `requires ... packages ... run pip install X` -> **Status:** `Missing Dep` -> **Suggestion:** "Run `pip install X`."
  * **Library Incompatibility:** `ImportError: cannot import name ... from transformers` -> **Status:** `Lib Version` -> **Suggestion:** "Check `transformers` version compatibility."
  * **Model Corrupt/Missing:** `Missing ... parameters` -> **Status:** `Model Error` -> **Suggestion:** "Re-download weights or check `trust_remote_code`."

### B. Enhanced Crash Reporting

* **Problem:** Stack traces in the Markdown table break readability.
* **Proposal:**
  * **Markdown:** Move full stack traces to a `<details>` block or a separate "Error Log" section at the bottom. Keep the table clean with just the "Status" (e.g., `❌ OOM`).
  * **JSON Artifact:** Save a `crash_report.json` for each failed model with full context (versions, args, stack trace) for automated analysis.

### C. Determinism & Sanity Checks

* **Problem:** Silent failures (NaNs) or race conditions aren't caught.
* **Proposal:**
  * **`--check-determinism <N>`:** Run generation N times. Flag if outputs differ.
  * **Logit Sanity:** Check for `NaN`/`Inf` in logits if accessible.

## 2. Quality Evaluation (Model Output)

The current "Output" column in `results.md` renders the table unreadable due to excessive width.

### A. Report Layout Overhaul

* **Problem:** The "Output" column makes the table scroll horizontally indefinitely.
* **Proposal:**
  * **Markdown:** Remove "Output" from the main metrics table. Create a **"Model Gallery"** section below the table where each model gets a header and a blockquote with its output.
  * **HTML:** Use a "Card" layout or a side-by-side comparison view instead of a single wide table.

### B. Improved Pattern Detection

* **Problem:** `mlx-community/paligemma2-10b-ft-docci-448-6bit` repeated "The scene is very visible..." ~20 times, but was only flagged as `context-ignored`.
* **Proposal:**
  * **Phrase Repetition:** Update `_detect_repetitive_patterns` to catch repeated *phrases* within a line, not just repeated *lines*.
  * **Metric:** Calculate "Unique N-gram Ratio" and flag if it drops below a threshold.

### C. Reference-Based Metrics

* **Problem:** We rely solely on "bad pattern" detection.
* **Proposal:**
  * **`--reference <text>`:** Allow passing a ground truth string.
  * **Metrics:** Calculate BLEU/ROUGE and Semantic Similarity (using a lightweight embedding model) against the reference.

## 3. Summary of Proposed Changes

1. **Refactor `results.md` generation:** Separate metrics (table) from text output (gallery).
2. **Implement `ErrorClassifier`:** Parse stack traces for OOM, Missing Deps, etc.
3. **Enhance `QualityThresholds`:** Add phrase-level repetition detection.
4. **Add `--reference` support:** For calculating correctness metrics.
