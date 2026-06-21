# Generated Reports

This directory contains generated reports and artifacts from running `check_models.py`.

The script's default output location is now this directory, keeping the project root clean.

## Directory Structure

```text
output/
├── reports/                  # Human-readable report files
│   ├── results.html          # Styled HTML report (viewable in browser)
│   ├── results.md            # Mode-aware public run index
│   ├── model_selection.md    # Ranked caption/metadata shortlist
│   ├── model_gallery.md      # Complete per-model evidence gallery
│   ├── review.md             # Review digest grouped by owner and user bucket
│   ├── results.tsv           # Tab-separated values table (for spreadsheets)
│   └── diagnostics.md        # Upstream issue report (conditional)
├── issues/                   # Generated GitHub issue templates (conditional)
│   ├── issue_001_crash.md    # One per failure cluster
│   └── issue_002_harness.md  # One per harness issue
├── repro_bundles/            # JSON reproduction bundles per failed model
│   └── 20260417T...*.json    # Timestamped bundle with error + env details
├── results.jsonl             # JSON Lines report (machine-readable results)
├── run.json                  # Stable run-level metadata contract
├── results.history.jsonl     # Append-only per-run history for regressions
├── check_models.log          # Detailed execution log
└── environment.log           # Complete Python environment dump
```

## Files

### Production Output (committed to git)

**In `reports/`** (human-readable):

- `reports/results.html` - Styled HTML report (viewable in browser)
- `reports/results.md` - Mode-aware public run index. In triage mode it suppresses cataloging
  and keyword scores and points to model-selection and diagnostics artifacts.
- `reports/model_selection.md` - Ranked shortlist for brief-caption and structured
  title/description/keywords decisions. This is not the complete run; it is a
  selection aid that points to the complete gallery.
- `reports/model_gallery.md` - Complete evidence-only record for every attempted
  model, including generated outputs, diagnostics, timing/memory summaries, and
  full per-model sections.
- `reports/review.md` - Review digest grouped by owner and user bucket
- `reports/results.tsv` - Tab-separated values table (for spreadsheets/data analysis)
- `reports/diagnostics.md` - Upstream issue report (only generated when failures/harness issues detected)

**In `output/`** (machine-readable and logs):

- `results.jsonl` - JSON Lines report (machine-readable results)
- `run.json` - Stable run-level machine contract with resolved mode, grounding policy, counts,
  versions, and artifact paths
- `results.history.jsonl` - Append-only per-run history for regressions/recoveries
- `check_models.log` - Detailed execution log from production benchmark runs
- `environment.log` - Complete Python environment dump (for reproducibility)

**Conditional subdirectories** (generated when failures are present):

- `issues/` - Ready-to-file GitHub issue markdown for clustered crashes and harness problems
- `repro_bundles/` - JSON reproduction bundles per failed model, containing error details, CLI args, and environment for reproducibility. Automatically pruned after 90 days (configurable via `--prune-repro-days`).

These files are regenerated each time you run the tool (**history is append-only**) and **committed to version control** so results are visible in GitHub.

### Test/Debug Output (excluded from git)

When running tests, integration checks, or debug runs, pass custom output paths
via CLI flags and use the `test_` prefix to avoid polluting production
results:

```bash
# Example: Running tests with debug-specific output
python -m check_models \
  --output-html output/test_results.html \
  --output-markdown output/test_results.md \
  --output-model-selection output/test_model_selection.md \
  --output-gallery-markdown output/test_model_gallery.md \
  --output-tsv output/test_results.tsv \
  --output-jsonl output/test_results.jsonl \
  --output-run-json output/test_run.json \
  --output-log output/test_check_models.log \
  [other options...]
```

Files matching the `test_*` prefix in `output/`, `reports/`, `issues/`, and
`repro_bundles/` are automatically excluded from git tracking via `.gitignore`.
The production Markdown reports are linted by the quality gate because they are
tracked artifacts. Keep generated markdownlint guards stable in
`check_models.py`, and do not commit ad-hoc debug output or one-off local run
churn unless it is an intentional benchmark snapshot.

**Separation strategy**:

- **Production runs**: Use default outputs (reports in `reports/`, machine data and logs in `output/`) → committed to git
- **Integration tests**: Use `test_cli_integration.{log,html,md,tsv,jsonl}` plus any test gallery artifact → gitignored (handled automatically by test suite)
- **Manual test/debug runs**: Pass custom paths with the `test_` prefix so
  `.gitignore` excludes the local artifacts
- **Pre-push git hook**: Runs pytest which uses test-specific output files → doesn't overwrite production log

This ensures your production benchmark results in git remain clean and only reflect intentional benchmark runs, not test/debug executions.

## CLI Logging Output

When you run `check_models.py`, the script provides rich terminal output with color-coded formatting and visual indicators:

### Startup Phase

```console
╔══════════════════════════════════════════╗
║ MLX Vision Language Model Check          ║
╚══════════════════════════════════════════╝

--- Library Versions ---
mlx:            0.29.4.dev
mlx-vlm:        0.3.2
Pillow:         12.0.0
...
Generated: 2025-10-31 15:30:45 PDT

--- System Information ---
macOS:        v15.1
Python:       v3.13.7

Hardware:
• Chip:        Apple M3 Max
• RAM:         96.0 GB
• CPU Cores:   14
• GPU Cores:   40
```

### Image Metadata Phase

If the image contains EXIF data:

```console
📷 Image Metadata (photo.jpg)
────────────────────────────────────
Make:           Apple
Model:          iPhone 15 Pro
DateTime:       2025:10:31 12:00:00
GPS Latitude:   37.7749° N
GPS Longitude:  122.4194° W
```

### Model Processing Phase

For each model tested:

```console
✓ qwen2-vl-7b-instruct               [Success]
  Tokens: 150 | Gen: 42 | TPS: 85.3 | Time: 2.5s
  Output: "A beautiful sunset over..."

✗ llava-1.5-7b-hf                    [Failed]
  Error: Model load timeout (60s exceeded)
```

**Visual Indicators:**

- `✓` (green) - Successful inference
- `✗` (red) - Failed inference
- Color-coded output:
  - **Green** - Success messages
  - **Yellow** - Warnings (e.g., missing EXIF data)
  - **Red** - Errors and failures
  - **Blue** - Informational messages (metadata, GPS coordinates)

### Summary Table

After all models complete, a formatted table displays results:

```console
╔════════════════════════════════════════════════════════════════╗
║                      Model Performance Summary                  ║
╚════════════════════════════════════════════════════════════════╝

Model                        Tokens  Gen  TPS    Time   Output
───────────────────────────────────────────────────────────────────
✓ qwen2-vl-7b-instruct         150   42  85.3   2.5s   A beautiful...
✓ llava-v1.6-mistral-7b-hf     180   38  72.1   3.1s   The image...
✗ llava-1.5-7b-hf               -     -    -      -    [Timeout]

─────────────────────────────────────────────────────
Overall runtime: 45.2s
Success rate: 2/3 (66.7%)
```

**Column Descriptions:**

- **Model** - Model identifier with success/fail indicator
- **Tokens** - Total prompt tokens processed
- **Gen** - Generation tokens produced
- **TPS** - Tokens per second (generation speed)
- **Time** - Total inference time
- **Output** - Model response (truncated for display)

### Finalization Phase

```console
💾 Reports generated:
   • HTML: src/output/reports/results.html
   • Markdown: src/output/reports/results.md
   • Model Selection: src/output/reports/model_selection.md
   • Gallery: src/output/reports/model_gallery.md
   • TSV: src/output/reports/results.tsv
  • JSONL: src/output/results.jsonl
  • Run JSON: src/output/run.json

⏱️  Total runtime: 45.2 seconds
```

## Log Levels

The script supports different verbosity levels via the `--verbose` flag:

- **Default** - Standard progress messages (INFO level)
- **`--verbose`** - Detailed diagnostic information including DEBUG-level messages (EXIF parsing, internal state, model loading details)

Example:

```bash
python -m check_models --verbose
```

## Log File Output

When using `--output-log`, all console output is also saved to a file without color codes:

```bash
python src/check_models.py --output-log src/output/results.log
```

The log file contains the same information as terminal output but in plain text format, suitable for:

- Post-execution analysis
- CI/CD pipeline integration
- Archiving run history
- Debugging issues
