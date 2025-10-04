# CLI Output Formatting Review & Improvement Suggestions

**Date**: 2025-10-03
**File**: `src/check_models.py`
**Purpose**: Review current formatting and colorization, suggest improvements for clarity and visual scanning

---

## Current State Analysis

### Color Scheme

The script uses a well-defined color palette:

```python
- RED (91)     : Errors, failures
- GREEN (92)   : Success indicators, completed tasks
- YELLOW (93)  : Warnings, processing status
- BLUE (94)    : Separators, structural elements
- MAGENTA (95) : Section headers, model names, important identifiers
- CYAN (96)    : Generated text output, library versions
- WHITE (97)   : Metrics, performance data
- GRAY (90)    : Debug messages
- BOLD         : Emphasis on headers and critical info
```

### Current Output Structure

#### Standard Mode (Non-Verbose)

1. **Section Headers**: Bold magenta with blue separator lines
2. **Processing Status**: Model name in magenta
3. **Summary Line**: Green/red based on success, with key metrics
4. **Generated Text Preview**: Cyan color

#### Verbose Mode

1. All of standard mode, plus:
2. **Detailed Metrics**: White text for numbers
3. **Token Summary**: Separate lines for tokens and TPS
4. **Timing Details**: Total, generation, and load times
5. **Memory Metrics**: Active, cached, and peak memory

---

## Issues & Opportunities for Improvement

### 1. **MLX-VLM Output Interleaving** ⚠️ HIGH PRIORITY

**Problem**: When `verbose=True` is passed to `mlx_vlm.generate()`, the library outputs its own progress information (token-by-token generation, timing info) directly to stdout/stderr. This output is **not** styled or prefixed by check_models.py, making it hard to distinguish script output from library output.

**Example of Confusion**:

```
[ PROCESSING MODEL: PHI-3-VISION ]
Processing 'test.jpg' with model: microsoft/Phi-3-vision-128k-instruct
Fetching 6 files: 100%|█████████████| 6/6 [00:00<00:00, 42.12it/s]
Prompt: 125 tokens, 1.234 tok/s
Generation: 78 tokens, 23.456 tok/s    ← MLX-VLM output (not styled)
✓ SUCCESS: Phi-3-vision-128k-instruct  ← check_models output (styled)
```

**Suggested Solutions**:

**Option A: Visual Prefix/Indent** (Easiest, Non-Intrusive)

- Add a visual indicator to check_models output to make it stand out
- Prefix check_models messages with `📊` or `▶` symbols
- Keep MLX-VLM output unmodified but clearly separated

```python
def print_cli_section(title: str) -> None:
    """Print a formatted CLI section header."""
    width = get_terminal_width(max_width=100)
    safe_title = title if "\x1b[" in title else title.upper()
    # Add visual prefix to make check_models output distinct
    logger.info("▶ [ %s ]", Colors.colored(safe_title, Colors.BOLD, Colors.MAGENTA))
    log_rule(width, char="━", color=Colors.BLUE, bold=False)
```

**Option B: Bracketed Context** (More Invasive)

- Wrap each model processing in clear "begin/end" markers
- Makes it obvious what's MLX-VLM vs what's check_models

```
┌─────────────────────────────────────────────────────
│ MODEL: microsoft/Phi-3-vision-128k-instruct
└─────────────────────────────────────────────────────
  ↓ MLX-VLM Output (unformatted):
    Fetching files...
    Prompt: 125 tokens, 1.234 tok/s
    Generation: 78 tokens, 23.456 tok/s
  ↑ End MLX-VLM Output
┌─────────────────────────────────────────────────────
│ ✓ check_models RESULT: SUCCESS
│   Metrics: total=4.13s gen=3.03s load=1.09s
└─────────────────────────────────────────────────────
```

**Option C: Capture and Reformat** (Most Control, Most Complex)

- Capture MLX-VLM's stdout/stderr in verbose mode
- Parse and reformat with consistent styling
- Requires stdout/stderr redirection during `generate()` call

```python
import io
import contextlib

def _run_model_generation_with_captured_output(...):
    if verbose:
        # Capture MLX-VLM's output
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        with contextlib.redirect_stdout(captured_stdout), \
             contextlib.redirect_stderr(captured_stderr):
            output = generate(...)

        # Re-emit with styling
        for line in captured_stdout.getvalue().splitlines():
            logger.info("  %s", Colors.colored(f"[mlx-vlm] {line}", Colors.GRAY))
    else:
        output = generate(...)
```

**Recommendation**: Start with **Option A** (visual prefixes) for immediate improvement, consider Option C for future enhancement.

---

### 2. **Inconsistent Metric Formatting**

**Problem**: Metrics use different formats in different contexts:

- Summary line: `tokens=1637 gen_tps=114`
- Verbose detailed: `Prompt Tokens: 1,488` (with thousands separator)
- Compact metrics: `tokens(total/prompt/gen)=1637/1488/149`

**Suggestion**: Standardize number formatting:

```python
# Current: Inconsistent
"tokens=1637"                    # No separator
"Prompt Tokens: 1,488"          # Comma separator
"tokens(total/prompt/gen)=1637/1488/149"  # Compact form

# Proposed: Consistent
"tokens=1,637"                   # Always use separator for > 999
"Prompt Tokens: 1,488"          # Keep separator
"tokens(total/prompt/gen)=1,637/1,488/149"  # Compact with separators
```

---

### 3. **Visual Hierarchy in Verbose Mode**

**Problem**: Detailed metrics use inconsistent indentation and headers aren't visually distinct enough.

**Current**:

```
✓ SUCCESS: Phi-3-vision-128k-instruct
Generated Text: This is a test image...
  Tokens: total=1,637 prompt=1,488 gen=149
  Generation TPS: 114
  Total Time: 4.13s
  Generation Time: 3.03s
  Model Load Time: 1.09s
  Performance Metrics:
    Time: 3.03s
    Memory (Active Δ): 0.5GB
    Memory (Cache Δ): 4.8GB
    Memory (Peak): 5.5GB
    Prompt Tokens: 1,488
    Generation Tokens: 149
    Prompt TPS: 421
```

**Suggested Improvement**:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ SUCCESS: Phi-3-vision-128k-instruct
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Generated Text:
   This is a test image showing a cat on a windowsill with sunlight
   streaming through, creating interesting shadows and highlights...

📊 Performance Summary:
   ├─ Total Time:      4.13s
   ├─ Generation:      3.03s (73%)
   └─ Model Load:      1.09s (27%)

🔢 Token Statistics:
   ├─ Prompt:          1,488 tokens @ 421 tok/s
   ├─ Generated:         149 tokens @ 114 tok/s
   └─ Total:           1,637 tokens

💾 Memory Usage:
   ├─ Active Delta:    0.5 GB
   ├─ Cache Delta:     4.8 GB
   └─ Peak:            5.5 GB
```

Implementation:

```python
def _log_verbose_success_details_mode(res: PerformanceResult, *, detailed: bool) -> None:
    """Emit verbose block with improved hierarchy."""
    if not res.generation:
        return

    # Success separator
    width = get_terminal_width(max_width=100)
    log_rule(width, char="━", color=Colors.GREEN, bold=True)

    # Generated text with emoji prefix
    gen_text = getattr(res.generation, "text", None) or ""
    logger.info("")  # Blank line for breathing room
    logger.info("📝 %s", Colors.colored("Generated Text:", Colors.BOLD, Colors.CYAN))
    _log_wrapped_label_value("   ", gen_text, color=Colors.CYAN)

    logger.info("")
    logger.info("📊 %s", Colors.colored("Performance Summary:", Colors.BOLD, Colors.WHITE))
    _log_tree_metrics(res)

    # ... rest of metrics with tree structure
```

---

### 4. **Compact Metrics Readability**

**Problem**: The single-line compact metrics can be hard to scan:

```
Metrics: total=4.13s gen=3.03s load=1.09s peak_mem=5.5GB tokens(total/prompt/gen)=1637/1488/149 gen_tps=114
```

**Suggestion**: Use clearer grouping and alignment:

```
Metrics: ⏱ total=4.13s gen=3.03s load=1.09s  💾 peak=5.5GB  🔢 tok=1,637 (p:1,488 g:149)  ⚡ tps=114
```

Or split into logical groups:

```
⏱  Time: total=4.13s  gen=3.03s  load=1.09s
💾  Memory: peak=5.5GB
🔢  Tokens: 1,637 (prompt:1,488  generated:149)  TPS: 114
```

---

### 5. **Error Messages**

**Current**: Error wrapping is good, but could be more visually distinct:

```
✗ FAILED: microsoft/Phi-3-vision-128k-instruct
Stage: processing
Error: Model loading failed: [Errno 2] No such file or directory: ...
```

**Suggestion**: Add error context boxes:

```
╔════════════════════════════════════════════════════════════════════
║ ✗ FAILED: microsoft/Phi-3-vision-128k-instruct
╠════════════════════════════════════════════════════════════════════
║ Stage:  processing
║ Error:  Model loading failed: [Errno 2] No such file or directory
║
║         Full traceback available with --verbose
╚════════════════════════════════════════════════════════════════════
```

---

### 6. **Section Separators**

**Current**: Uses simple dashes, which can get lost in output:

```
-------------------------------------------------
```

**Suggestion**: Use unicode box-drawing characters for better visibility:

```python
def print_cli_separator() -> None:
    """Print a visually distinct separator line."""
    width = get_terminal_width(max_width=100)
    # Use unicode box-drawing for better visual separation
    log_rule(width, char="─", color=Colors.BLUE, bold=False)

def log_rule(width: int, char: str = "─", color: str = "", bold: bool = False) -> None:
    """Log a horizontal rule with optional color."""
    colors = [c for c in [Colors.BOLD if bold else "", color] if c]
    logger.info(Colors.colored(char * width, *colors))
```

---

### 7. **Summary Table at End**

**Current**: The final summary table is good but could be enhanced:

**Suggestion**: Add visual indicators and color coding:

```
═══════════════════════════════════════════════════════════════════════
                     PERFORMANCE SUMMARY
═══════════════════════════════════════════════════════════════════════

| Model                          | Status | Tokens | Gen TPS | Peak Mem | Time   |
|--------------------------------|--------|--------|---------|----------|--------|
| ✓ Phi-3-vision                 |   OK   | 1,637  |   114   |  5.5 GB  | 4.13s  |
| ✓ llava-1.5-7b                 |   OK   | 2,241  |   89    |  7.2 GB  | 5.82s  |
| ✗ pixtral-12b                  |  FAIL  |   -    |    -    |    -     |   -    |
|--------------------------------|--------|--------|---------|----------|--------|
| Total Models: 3                | Pass: 2  Fail: 1                               |
═══════════════════════════════════════════════════════════════════════

⏱  Total Runtime: 12.45s
📊 HTML Report: /path/to/results.html
📝 Markdown Report: /path/to/results.md
```

---

## Implementation Priority

### Phase 1: Quick Wins (Low Risk, High Impact)

1. ✅ Add visual prefixes (▶, 📊, 📝, etc.) to check_models output
2. ✅ Improve separator lines with unicode box-drawing characters
3. ✅ Standardize number formatting with thousands separators
4. ✅ Add blank lines for breathing room between sections

### Phase 2: Moderate Changes (Medium Risk, High Impact)

5. 🔄 Restructure verbose output with tree-style hierarchy
6. 🔄 Add emoji icons to section headers for quick scanning
7. 🔄 Improve error message boxes with border characters
8. 🔄 Enhanced summary table with visual indicators

### Phase 3: Advanced Features (Higher Risk, High Impact)

9. 🔮 Capture and reformat MLX-VLM output for consistent styling
10. 🔮 Add progress indicators for multi-model runs
11. 🔮 Support for alternate output formats (JSON, CSV) alongside pretty printing

---

## Code Changes Required

### Minimal Changes (Phase 1)

```python
# 1. Update print_cli_section to add prefix
def print_cli_section(title: str) -> None:
    """Print a formatted CLI section header with visual prefix."""
    width = get_terminal_width(max_width=100)
    safe_title = title if "\x1b[" in title else title.upper()
    # Add ▶ prefix to make check_models output visually distinct
    logger.info("▶ [ %s ]", Colors.colored(safe_title, Colors.BOLD, Colors.MAGENTA))
    log_rule(width, char="─", color=Colors.BLUE, bold=False)

# 2. Update log_rule to use better characters
def log_rule(width: int, char: str = "─", color: str = "", bold: bool = False) -> None:
    """Log a horizontal rule with optional color."""
    colors = [c for c in [Colors.BOLD if bold else "", color] if c]
    logger.info(Colors.colored(char * width, *colors))

# 3. Update print_cli_separator
def print_cli_separator() -> None:
    """Print a visually distinct separator line."""
    width = get_terminal_width(max_width=100)
    log_rule(width, char="─", color=Colors.BLUE, bold=False)

# 4. Ensure fmt_num always includes thousands separator
def fmt_num(value: float | int | None, decimals: int = 0) -> str:
    """Format numeric values with thousands separators."""
    if value is None:
        return "-"
    if isinstance(value, float):
        if decimals > 0:
            return f"{value:,.{decimals}f}"
        return f"{value:,.0f}"
    return f"{value:,}"

# 5. Add breathing room in verbose output
def _log_verbose_success_details_mode(res: PerformanceResult, *, detailed: bool) -> None:
    """Emit verbose block with improved spacing."""
    if not res.generation:
        return

    # Add blank lines for visual breathing room
    logger.info("")

    gen_text = getattr(res.generation, "text", None) or ""
    _log_wrapped_label_value("📝 Generated Text:", gen_text, color=Colors.CYAN)

    logger.info("")  # Breathing room

    if detailed:
        logger.info("📊 %s", Colors.colored("Performance Metrics:", Colors.BOLD, Colors.MAGENTA))
        _log_token_summary(res)
        _log_detailed_timings(res)
        logger.info("")
        _log_perf_block(res)
    else:
        _log_compact_metrics(res)
```

---

## Testing Recommendations

1. **Visual Regression Testing**: Take screenshots of output before/after changes
2. **Terminal Compatibility**: Test on various terminals (iTerm2, Terminal.app, VS Code terminal)
3. **NO_COLOR Support**: Verify output remains readable with `NO_COLOR=1`
4. **Width Testing**: Test with narrow (80 col) and wide (200 col) terminals
5. **Verbose vs Standard**: Ensure both modes remain clear and distinct

---

## Backward Compatibility

All suggested changes maintain backward compatibility:

- Color codes only appear if terminal supports them (TTY detection + NO_COLOR)
- Unicode characters degrade gracefully in terminals that don't support them
- Log messages remain parseable for automated tools
- HTML/Markdown output unaffected (uses separate formatting)

---

## Summary of Key Recommendations

🏆 **Top Priority**: Add visual prefixes/markers to distinguish check_models output from MLX-VLM output

📊 **High Impact**: Improve verbose mode hierarchy with tree-style formatting and emoji icons

🎨 **Quick Win**: Better separator lines using unicode box-drawing characters

🔢 **Consistency**: Standardize number formatting with thousands separators throughout

⚠️ **Future**: Consider capturing MLX-VLM output for consistent styling in verbose mode
