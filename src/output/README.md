# Generated Reports

This directory contains generated HTML, Markdown, and TSV reports from running `check_models.py`.

The script's default output location is now this directory, keeping the project root clean.

## Files

### Production Output (committed to git)

- `results.html` - Styled HTML report (viewable in browser)
- `results.md` - GitHub-friendly Markdown report
- `results.tsv` - Tab-separated values table (for spreadsheets/data analysis)
- `check_models.log` - Detailed execution log from production benchmark runs

These files are regenerated each time you run the tool and **committed to version control** so results are visible in GitHub.

### Test/Debug Output (excluded from git)

When running tests, integration checks, or debug runs, pass custom output paths via CLI flags to avoid polluting production results:

```bash
# Example: Running tests with debug-specific output
python check_models.py \
  --output-html output/test_results.html \
  --output-markdown output/test_results.md \
  --output-tsv output/test_results.tsv \
  --output-log output/test_check_models.log \
  [other options...]
```

Files matching `test_*.{html,md,tsv,log}` are automatically excluded from git tracking via `.gitignore`.

**Separation strategy**:

- **Production runs**: Use default outputs (`check_models.log`, `results.html`, `results.md`, `results.tsv`) → committed to git
- **Integration tests**: Use `test_cli_integration.{log,html,md,tsv}` → gitignored (handled automatically by test suite)
- **Manual test/debug runs**: Pass custom paths with `test_` prefix or any other name except the production defaults → gitignored
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
   • HTML: src/output/results.html
   • Markdown: src/output/results.md

⏱️  Total runtime: 45.2 seconds
```

## Log Levels

The script supports different verbosity levels via `--log-level`:

- `DEBUG` - Detailed diagnostic information (EXIF parsing, internal state)
- `INFO` - Standard progress messages (default)
- `WARNING` - Warning messages (missing metadata, deprecated features)
- `ERROR` - Error messages (model failures, file I/O issues)
- `CRITICAL` - Fatal errors that stop execution

Example:

```bash
python src/check_models.py --log-level DEBUG
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
