# Table Formatting Library Decision

<!-- markdownlint-disable MD032 MD031 MD036 -->

**Date:** October 17, 2025  
**Context:** Analysis of table formatting dependencies and custom HTML manipulation code  
**Decision:** Keep `tabulate` for table generation; maintain regex-based HTML manipulation

---

## Environment

- **Python Version:** 3.13.7
- **tabulate Version:** 0.9.0 (minimum required: >=0.9.0)
- **Project:** MLX Vision Language Model Benchmarking CLI Tool

---

## Background

During code review, questions arose about:

1. Whether we could simplify table generation by removing `tabulate` dependency
2. Whether custom HTML manipulation could be improved with a dedicated library

---

## Analysis: Replacing tabulate with Custom Code

### Attempt Summary

Initially attempted to replace `tabulate` with custom table generation functions to reduce dependencies.

### Custom Implementation

Created ~240 lines of custom table generators:
- `generate_html_table()` - 38 lines
- `generate_markdown_table()` - 88 lines (required complexity warning `noqa: PLR0912`)
- `generate_console_table()` - 68 lines (required complexity warning `noqa: PLR0912`)
- `_generate_fancy_grid_table()` - 46 lines

### What Was Replaced

Four simple `tabulate()` calls:
```python
# Console metadata display
table = tabulate(rows, headers=headers, tablefmt="fancy_grid", colalign=["left", "left"])

# CLI performance summary
table = tabulate(rows, headers=headers, tablefmt="plain", colalign=colalign, maxcolwidths=widths)

# HTML export
html_table = tabulate(rows, headers=headers, tablefmt="unsafehtml", colalign=colalign)

# Markdown export
markdown_table = tabulate(rows, headers=markdown_headers, tablefmt="pipe", colalign=colalign)
```

### Verdict: REVERTED

**Why we reverted:**
1. ❌ **Not simpler** - Added 240 lines vs 1 import + 4 simple calls
2. ❌ **Not easier to maintain** - Now responsible for column width calculation, alignment logic, edge cases
3. ⚠️ **Functionality risk** - Custom code likely has bugs in edge cases that tabulate already handles
4. ❌ **Complexity warnings** - Two functions required `noqa` directives for excessive branching

**Why tabulate is the right choice:**
1. ✅ **Mature library** - Handles edge cases (empty cells, varying widths, Unicode, etc.)
2. ✅ **Well-tested** - Used by thousands of projects
3. ✅ **Minimal API** - Simple function calls, not complex to maintain
4. ✅ **Multiple formats** - Supports 20+ table formats (we use 4)
5. ✅ **Active maintenance** - Regular updates and bug fixes

---

## Analysis: Current Use of tabulate

### What We Use tabulate For

| Feature | Usage | Location |
|---------|-------|----------|
| `tablefmt="fancy_grid"` | EXIF metadata display with Unicode borders | `print_exif_summary()` |
| `tablefmt="plain"` | CLI performance summary (no borders) | `print_model_stats()` |
| `tablefmt="unsafehtml"` | HTML report generation | `generate_html_report()` |
| `tablefmt="pipe"` | GitHub-flavored Markdown export | `generate_markdown_report()` |
| `colalign` | Column alignment (left/right) | All 4 calls |
| `maxcolwidths` | Column width constraints | CLI summary only |

### Custom Embellishments (Built on Top of tabulate)

These are **genuinely necessary** and not duplicating tabulate functionality:

1. **Console Decorations** (lines 1620-1633)
   - Adding title headers with `log_rule()` borders
   - Splitting table into lines for line-by-line logger output
   - **Reason:** tabulate returns a string; we need custom framing

2. **HTML Post-Processing** (lines 2004-2024)
   - `_mark_failed_rows_in_html()` - Adding `class="failed-row"` to failed test rows
   - `_build_full_html_document()` - Wrapping table in complete HTML with CSS
   - **Reason:** tabulate doesn't support CSS classes or full document generation

3. **Markdown Post-Processing** (lines 2327-2410)
   - `_escape_markdown_in_text()` - Escaping pipes and HTML tags in user content
   - `_escape_markdown_diagnostics()` - Aggressive escaping for error messages
   - `normalize_markdown_trailing_spaces()` - Cleaning up trailing whitespace
   - **Reason:** tabulate doesn't escape user content that could break table structure

4. **Data Preprocessing**
   - ANSI color code insertion into cells/headers
   - Multi-line text handling (converting newlines to `<br>`)
   - Success/fail indicators (`✓`/`✗`)
   - **Reason:** tabulate passes through cell content as-is (correct behavior)

### What tabulate Does NOT Provide

- ❌ HTML element class/attribute manipulation
- ❌ Content escaping (Markdown pipes, HTML tags)
- ❌ Full HTML document generation
- ❌ ANSI color code management
- ❌ Custom borders/decorations outside the table itself

**Conclusion:** We're using tabulate appropriately. Our embellishments are genuine requirements, not duplicated functionality.

---

## Analysis: HTML Manipulation Libraries

### Current Approach

Manual string manipulation and regex for HTML modifications:

```python
def _mark_failed_rows_in_html(html_table: str, results: list[PerformanceResult]) -> str:
    """Add class="failed-row" to <tr> elements whose corresponding result failed."""
    # 21 lines using re.sub() and string slicing
    # Single use case: add CSS class to specific table rows
```

### Alternative Libraries Considered

#### 1. BeautifulSoup4

**Version evaluated:** 4.12.x (current stable)

**Pros:**
- ✅ Clean, Pythonic API for HTML manipulation
- ✅ Proper HTML parsing (safer than regex)
- ✅ Easy element modification: `tr['class'] = ['failed-row']`
- ✅ Handles edge cases (existing classes, malformed HTML)
- ✅ Industry standard for HTML manipulation

**Cons:**
- ❌ Additional dependency (`beautifulsoup4`)
- ❌ Requires parser: either stdlib `html.parser` or `lxml` (C dependency)
- ❌ May reformat HTML output (whitespace, attribute order)
- ❌ Overkill for **single 21-line function**

**Code comparison:**

```python
# Current regex (21 lines, works)
body_html = re.sub(r"<tr>", _row_replacer, body_html)

# With BeautifulSoup4 (~15 lines, cleaner)
soup = BeautifulSoup(html_table, 'html.parser')
tbody = soup.find('tbody')
for idx, tr in enumerate(tbody.find_all('tr', recursive=False)):
    if idx in failed_set:
        tr['class'] = tr.get('class', []) + ['failed-row']
return str(soup)
```

#### 2. lxml

**Version evaluated:** 5.x

**Pros:**
- ✅ Very fast HTML/XML parsing
- ✅ XPath support for complex queries
- ✅ Can modify elements

**Cons:**
- ❌ C library dependency (platform-specific builds)
- ❌ More complex API than BeautifulSoup
- ❌ Overkill for our minimal needs

#### 3. html.parser (Python stdlib)

**Built-in module**

**Pros:**
- ✅ No additional dependency
- ✅ Can parse and build HTML

**Cons:**
- ❌ More verbose than regex for simple replacements
- ❌ Lower-level API (requires subclassing `HTMLParser`)
- ❌ Not simpler than current regex approach

#### 4. bleach

**Version evaluated:** 6.x

**Pros:**
- ✅ Designed for HTML sanitization

**Cons:**
- ❌ Additional dependency
- ❌ Focused on security sanitization (not manipulation)
- ❌ Not designed for our use case

---

## Decision: Keep Current Regex Approach

### Rationale

1. **Minimal use case** - Only ONE function (`_mark_failed_rows_in_html`) manipulates HTML
2. **Works correctly** - Current regex approach is tested and not buggy
3. **Dependency minimization** - Avoiding extra dependencies aligns with project goals
4. **Code locality** - The 21-line function is well-commented, localized, and maintainable
5. **Not error-prone** - Simple pattern matching on `<tr>` tags with clear logic

### When to Reconsider

Add BeautifulSoup4 if ANY of these become true:
- We need to manipulate HTML in 3+ different places
- The regex approach starts causing bugs
- We need more complex HTML manipulation (nested elements, attributes)
- We add other features requiring HTML parsing

### Documentation of Current Approach

The HTML manipulation is localized in two functions:

1. **`_mark_failed_rows_in_html()`** (line 2004)
   - Purpose: Add CSS class to failed test rows
   - Technique: Regex replacement within `<tbody>` section
   - Complexity: 21 lines, straightforward logic

2. **`_escape_html_tags_selective()`** (line 527)
   - Purpose: Escape HTML-like tags except GitHub-allowed safe tags
   - Technique: Regex pattern matching with allowlist
   - Complexity: 18 lines, well-tested patterns

Both functions:
- ✅ Are well-commented
- ✅ Handle edge cases (missing tbody, no failed rows)
- ✅ Have clear, single responsibilities
- ✅ Use simple, readable regex patterns
- ✅ Are tested by integration tests

---

## Dependency Strategy

### Current Runtime Dependencies (6 total)

From `pyproject.toml` [project.dependencies]:

```toml
dependencies = [
    "mlx>=0.29.1",           # Core ML framework
    "mlx-vlm>=0.0.9",        # Vision-language models
    "Pillow>=10.3.0",        # Image processing
    "huggingface-hub>=0.23.0", # Model cache/discovery
    "tabulate>=0.9.0",       # Table formatting ← THIS DECISION
    "tzlocal>=5.0",          # Timezone handling
]
```

### Strategy

**Keep dependencies minimal but pragmatic:**
- ✅ Use well-maintained libraries for complex domains (table formatting)
- ✅ Avoid reinventing wheels that are hard to get right (column alignment, width calculation)
- ❌ Don't add dependencies for single-function use cases (HTML manipulation)
- ❌ Don't add dependencies that we can easily implement ourselves (simple regex)

---

## Future Considerations

### If Table Formatting Needs Grow

If we need features beyond tabulate's capabilities:
1. Check if newer tabulate versions support it
2. Consider `rich` library (modern terminal formatting, supports colors natively)
3. Consider `prettytable` (more table customization options)

### If HTML Manipulation Needs Grow

If we need to manipulate HTML in multiple places:
1. Add `beautifulsoup4>=4.12.0` to dependencies
2. Use `html.parser` backend (no C dependencies)
3. Refactor HTML manipulation into a dedicated module
4. Add unit tests specifically for HTML manipulation

---

## Testing Coverage

### Current State

- ✅ All 29 integration tests pass with `tabulate`
- ✅ Tests cover Markdown table generation (verify pipe escaping works)
- ✅ Tests cover HTML report generation (verify output file creation)
- ❌ No dedicated unit tests for `_mark_failed_rows_in_html()` (covered by integration)
- ❌ No dedicated unit tests for HTML escaping functions (covered by integration)

### Recommendations

If we see bugs in HTML manipulation:
1. Add unit tests for `_mark_failed_rows_in_html()` with various failure patterns
2. Add unit tests for HTML escaping with edge cases (nested tags, malformed HTML)
3. Then consider moving to BeautifulSoup4 for robustness

---

## References

### Library Documentation

- **tabulate:** <https://github.com/astanin/python-tabulate>
  - Version used: 0.9.0
  - Last updated: 2022 (stable, mature)
  - License: MIT

- **BeautifulSoup4:** <https://www.crummy.com/software/BeautifulSoup/>
  - Version evaluated: 4.12.x
  - Last updated: 2024 (actively maintained)
  - License: MIT

### Related Project Documentation

- `docs/IMPLEMENTATION_GUIDE.md` - Dependency management strategy
- `docs/notes/COMPREHENSIVE_REVIEW_2025_10.md` - Overall code review findings
- `README.md` - Dependency list and rationale

---

## Summary

**Decision:** ✅ Keep `tabulate` for table generation; ✅ Keep regex for minimal HTML manipulation

**Key Insights:**
1. Removing `tabulate` would add 240 lines of complex code for minimal benefit
2. Current HTML manipulation is appropriate for its single-function use case
3. BeautifulSoup4 would be worthwhile if we had 3+ HTML manipulation functions
4. Dependency minimization should be pragmatic, not dogmatic

**Action Items:**
- ✅ Document this decision (this file)
- ✅ No code changes needed
- ⏭️ Future: Monitor if HTML manipulation needs grow beyond single function
- ⏭️ Future: If bugs appear in HTML regex, reconsider BeautifulSoup4

---

**Reviewed by:** GitHub Copilot  
**Approved by:** [Maintainer approval pending]
