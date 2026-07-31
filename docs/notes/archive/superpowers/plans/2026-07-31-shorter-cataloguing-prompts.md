# Shorter Cataloguing Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shorten the default blind and assisted catalogue prompts while presenting existing descriptive metadata explicitly as fallible hints.

**Architecture:** Keep prompt construction and prompt-aware diagnostics in `src/check_models.py`. Generate context before the response contract so the prompt ends with the required output labels, and extend existing context/draft parsers to recognise the new labels without breaking analysis of retained prompts that use the historical `Existing …` wording.

**Tech Stack:** Python 3.13, pytest, Ruff, existing `check_models.py` helpers.

## Global Constraints

- Keep the triage prompt exactly `Describe this image briefly.`
- Use `Context: Descriptive hints:` when descriptive hints are the first context block and `Descriptive hints:` after authoritative context.
- Use `Title hint:`, `Description hint:`, and `Keyword hints:` field labels.
- Keep historical `Existing title:`, `Existing description:`, and `Existing keywords:` prompt analysis compatible.
- Run Python and Make commands only after activating the `mlx-vlm` conda environment.

---

### Task 1: Concise default prompt contracts

**Files:**

- Modify: `src/check_models.py`
- Test: `src/tests/test_exif_extraction.py`

**Interfaces:**

- Consumes: `_build_metadata_provenance(metadata)` and existing prompt-field compactors.
- Produces: `_build_cataloguing_prompt(metadata, include_metadata_hints=True) -> str` with concise blind and assisted prompt text.

- [ ] **Step 1: Write failing prompt-behaviour tests**

Assert that the blind prompt contains no context-only claims, stays within a compact word budget, and ends with the three output labels. Assert that the assisted prompt uses the new hint heading and field labels, treats hints as fallible, places context before `Write:`, and ends with the same response schema.

- [ ] **Step 2: Run the focused tests and verify RED**

Run `pytest src/tests/test_exif_extraction.py -q` and confirm failures identify the old long prompt and historical labels.

- [ ] **Step 3: Implement the concise prompt builder**

Build the prompt in this order: concise evidence policy, authoritative/descriptive context blocks when present, three short field requirements, then:

```text
Return exactly these three sections and nothing else:
Title:
Description:
Keywords:
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run `pytest src/tests/test_exif_extraction.py -q` and confirm all tests pass.

### Task 2: Prompt-aware diagnostics compatibility

**Files:**

- Modify: `src/check_models.py`
- Test: `src/tests/test_quality_analysis.py`
- Test: `src/tests/test_pure_logic_functions.py`

**Interfaces:**

- Consumes: generated prompt text and historical retained prompt text.
- Produces: `_extract_prompt_context_text(...)` spanning adjacent authoritative/descriptive blocks and `_unchanged_draft_fields(...)` recognising both label generations.

- [ ] **Step 1: Write failing diagnostic-behaviour tests**

Use a generated assisted prompt to prove keyword hints remain available to overlap analysis and unchanged hint fields are detected. Add a literal historical prompt proving the former `Existing …` labels remain supported.

- [ ] **Step 2: Run the focused tests and verify RED**

Run `pytest src/tests/test_quality_analysis.py src/tests/test_pure_logic_functions.py -q` and confirm failures are caused by the parser not yet recognising the new heading and labels.

- [ ] **Step 3: Extend the existing parsers minimally**

Allow `Descriptive hints:` as the adjacent context section and match either `Title hint` / `Description hint` / `Keyword hints` or historical `Existing …` field labels.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run `pytest src/tests/test_quality_analysis.py src/tests/test_pure_logic_functions.py -q` and confirm all tests pass.

### Task 3: Documentation and verification

**Files:**

- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes: the completed prompt and parser changes.
- Produces: maintainer-facing release note and a fully verified branch.

- [ ] **Step 1: Update `[Unreleased]`**

Document that default catalogue prompts are shorter, descriptive metadata is explicitly labelled as hints, and retained historical prompts remain analysable.

- [ ] **Step 2: Run repository gates**

Run `make format`, `make -C src lint-fix`, `make lint`, and `make quality` in the activated conda environment.

- [ ] **Step 3: Inspect the final diff**

Run `git diff --check`, inspect `git diff`, and confirm no generated `src/output/` artifacts changed.
