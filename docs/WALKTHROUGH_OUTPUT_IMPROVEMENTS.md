# Walkthrough: Output Improvements

## Overview

This update enhances the `mlx-vlm-check` tool's output to be more effective for identifying bugs and evaluating model quality.

## Changes

### 1. Intelligent Error Classification

- **Feature**: Automatically categorizes errors into actionable types (e.g., `OOM`, `Missing Dep`, `Lib Version`, `Model Error`, `Timeout`).
- **Implementation**: Added `_classify_error` function using regex patterns.
- **Benefit**: Users can quickly see *why* a model failed without digging into stack traces immediately.

### 2. Enhanced Repetition Detection

- **Feature**: Detects repetitive *phrases* (n-grams) in addition to single tokens.
- **Implementation**: Updated `_detect_repetitive_output` to check for repeated 4-10 word sequences.
- **Benefit**: Catches "looping" behavior where a model repeats a sentence or phrase endlessly.

### 3. Report Layout Overhaul (Markdown)

- **Feature**: Separated the "Output" column from the main metrics table.
- **New Section**: Added a "Model Gallery" section below the table.
- **Benefit**:
  - The metrics table is now readable on GitHub (no horizontal scrolling).
  - Full model output is displayed in blockquotes, preserving formatting.
  - Quality warnings are clearly listed under each model's output.

### 4. Quality Analysis Improvements

- **Feature**: Added `issues` property to `GenerationQualityAnalysis`.
- **Benefit**: Centralized logic for aggregating human-readable issue descriptions.

## Verification

- **Unit Test**: Created and ran `src/tests/test_report_generation.py` (subsequently deleted) to verify the new Markdown layout.
- **Environment**: Verified tests run correctly in the `mlx-vlm` Conda environment.
- **Lints**: All code passes `ruff check` and `ruff format`.

## Example Output (Model Gallery)

```markdown
## Model Gallery

Full output from each model:

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-4bit

**Metrics:** 50.5 TPS | 100 tokens

```text
The image shows a cat sitting on a mat...
```

⚠️ **Quality Warnings:**

- Repetitive output (cat)

```
