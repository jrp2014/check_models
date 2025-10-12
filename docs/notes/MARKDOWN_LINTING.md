# Setting up Markdown Linting

The project uses `markdownlint-cli2` to ensure consistent markdown formatting across documentation files. This is **optional** but recommended for contributors who edit documentation.

## Why Markdown Linting?

- Ensures consistent formatting across all `.md` files
- Catches common markdown mistakes
- Enforces best practices (blank lines, heading structure, etc.)
- Runs automatically in `make quality` if available

## Installation

### Option 1: Install Node.js/npm (Recommended)

If you plan to contribute documentation frequently:

**macOS (via Homebrew):**

```bash
brew install node
```

**macOS (via official installer):**

Download from [nodejs.org](https://nodejs.org/)

**After installing Node.js:**

```bash
# Navigate to src/ directory
cd src/

# Install markdownlint-cli2
make install-markdownlint

# Or manually:
npm install
```

This installs `markdownlint-cli2` to `node_modules/.bin/`.

### Option 2: Use npx (No Installation)

If you have npm installed but don't want to install packages locally:

```bash
# Run markdownlint on-demand (downloads if needed)
npx markdownlint-cli2 '**/*.md'
```

The quality script automatically uses npx as a fallback if `markdownlint-cli2` is not installed locally.

### Option 3: Skip Markdown Linting

If you don't have Node.js/npm and don't want to install it:

- Markdown linting will be **automatically skipped** with a warning
- All other quality checks (ruff, mypy) will still run
- CI will still check markdown formatting

## Usage

### Automatic (via Make)

```bash
make quality          # Includes markdown linting if available
make quality-strict   # Requires markdown linting (fails if unavailable)
```

### Manual

```bash
# Local installation
npx markdownlint-cli2 '**/*.md'

# Or if installed via npm:
./node_modules/.bin/markdownlint-cli2 '**/*.md'

# Or via package.json script:
npm run lint:md
```

## Configuration

Markdown linting rules are configured in `.markdownlint-cli2.jsonc` (if present) or use defaults.

Common rules enforced:

- MD031: Fenced code blocks surrounded by blank lines
- MD032: Lists surrounded by blank lines
- MD022: Headings surrounded by blank lines
- MD025: Single top-level heading per document
- MD041: First line must be a top-level heading

## Troubleshooting

### "markdownlint not found" warning

This is normal if you haven't installed Node.js/npm. The warning is informational only.

**To resolve:**

1. Install Node.js (see Option 1 above)
2. Run `make install-markdownlint`
3. Re-run `make quality`

### npm install fails

```bash
# Clear npm cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### I don't want Node.js installed

That's fine! Markdown linting is optional for local development. CI will catch markdown issues if you submit a PR.

## CI Behavior

In CI/CD pipelines:

- If npx is available, markdown linting runs automatically
- If npx is unavailable, the build fails (CI requires full quality checks)
- Contributors without Node.js can rely on CI for markdown validation

## Further Reading

- [markdownlint-cli2 GitHub](https://github.com/DavidAnson/markdownlint-cli2)
- [Markdown linting rules](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md)
