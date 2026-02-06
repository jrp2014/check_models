# README Progressive Disclosure Restructuring

**Date**: 2025-01-10

## Objective

Restructure user-facing documentation (README.md) to follow progressive disclosure principles: make it quick for new developers to get started while preserving all important details, workarounds, and technical information by presenting them later in the text.

## Changes Made

### 1. Simplified Introduction

**Before**: Verbose technical description
**After**: One-line benefit-focused tagline

```markdown
Run and benchmark Vision-Language Models on Apple Silicon with MLX.
```

### 2. Condensed Quick Start

**Before**: Multiple sections with explanations
**After**: 3 essential commands with immediate feedback

```bash
make bootstrap     # One-time setup
make dev          # Install for development
make run          # Run the check (uses default model)
```

Added clear next step with link to detailed usage section.

### 3. Benefit-Focused "What You Get"

**Before**: Prose paragraph describing features
**After**: Scannable bullet-point list with emojis

- ✅ Instant feedback during model checks
- 📊 Performance metrics (tokens/sec, memory usage)
- 🎯 Multiple output formats (MD, HTML, JSON)
- 🛡️ Smart handling of failures
- 📝 Automatic metadata collection

### 4. Progressive Information Architecture

**New Structure**:

1. **Quick Start** (3 steps) - Get running immediately
2. **What You Get** (benefits) - Understand value proposition
3. **Detailed Usage** (common patterns) - Real examples when needed
4. **For Developers** (quick setup) - Contributor onboarding
5. **Using Make Commands** (essential commands) - Common workflows
6. **Advanced Topics** (all commands + details) - Comprehensive reference
7. **Repository Structure** - Project layout
8. **More Documentation** - Deep dives and guides

### 5. Preserved All Information

No content was removed - everything was reorganized:

- Basic usage → "Detailed Usage" section with code examples
- All make commands → "Advanced Topics" with full table
- Development setup → "For Developers" quick section + link to CONTRIBUTING.md
- Repository structure → Moved to end, still complete
- Dependencies → Moved to "Advanced Topics"
- Type stubs → Retained in developer section with examples

## Principles Applied

### Progressive Disclosure

Present information in layers:

1. **Essential** - What you need to get started (Quick Start)
2. **Benefit** - Why you should care (What You Get)
3. **Common** - Most frequent use cases (Detailed Usage)
4. **Reference** - Complete information (Advanced Topics)

### Scannability

- Used bullet points instead of prose
- Added emojis for visual anchors
- Short paragraphs with clear headings
- Code examples for common patterns

### Clear Navigation

- Explicit link from Quick Start to Detailed Usage
- Logical flow from beginner → intermediate → advanced
- Cross-references to other documentation

## Link Fragment Fix

Fixed markdown linting error MD051 by changing the link text to lowercase, which is more natural in the sentence context while maintaining the correct anchor reference.

## Validation

- ✅ All markdown lint errors cleared
- ✅ All information preserved
- ✅ Clear progression from beginner to advanced
- ✅ Quick start takes < 30 seconds to understand
- ✅ Details available but not blocking

## Next Steps

Consider applying the same progressive disclosure pattern to:

- `docs/CONTRIBUTING.md` - Developer onboarding
- `src/README.md` - CLI reference

The goal is consistent UX across all documentation where users can:

1. Get started quickly
2. Find common patterns easily
3. Access complete details when needed

## References

- Progressive Disclosure: <https://www.nngroup.com/articles/progressive-disclosure/>
- Documentation Best Practices: <https://documentation.divio.com/>
