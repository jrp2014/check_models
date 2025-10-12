# Local MLX Development Workflow

**Date**: 2025-10-12  
**Context**: Support for local MLX repository development

## Overview

The `update.sh` script now automatically detects and updates local MLX development repositories, making it easy to work with bleeding-edge versions or custom modifications of MLX, MLX-LM, and MLX-VLM.

## Directory Structure

The script expects local MLX repositories to be siblings of the `scripts` directory:

```text
/Users/jrp/Documents/AI/mlx/
├── mlx/                 # Core MLX framework
├── mlx-lm/              # MLX Language Models
├── mlx-vlm/             # MLX Vision-Language Models
└── scripts/             # This project (mlx-vlm-check)
    └── src/
        └── tools/
            └── update.sh
```

## How It Works

### Automatic Detection

When `update.sh` runs, it:

1. **Checks for local repositories** at `../../{mlx,mlx-lm,mlx-vlm}` relative to the script
2. **Verifies .git directories** to confirm they're git repositories
3. **Processes repositories in order**: mlx → mlx-lm → mlx-vlm (dependency order)

### Update Process Per Repository

For each detected repository:

1. **Git pull** - Updates the repository from its remote
2. **Special handling for mlx-vlm** - Installs `opencv-python` first (required dependency)
3. **Install requirements.txt** - If present, runs `pip install -U -r requirements.txt`
4. **Install package** - Runs `pip install -e .` (editable/development mode)

### Integration with Package Management

After updating local repos, the script:

- Sets `MLX_IS_LOCAL=1` flag
- Skips PyPI updates for MLX packages in `RUNTIME_PACKAGES`
- Continues with other dependencies (Pillow, huggingface-hub, etc.)

## Usage Examples

### Basic Update with Local Repos

```bash
cd /Users/jrp/Documents/AI/mlx/scripts/src/tools
./update.sh
```

Output:

```text
[update.sh] Checking for local MLX development repositories...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Updating local repository: mlx
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[update.sh] Running git pull in mlx...
✓ Git pull successful
[update.sh] No requirements.txt found
[update.sh] Installing mlx package...
✓ mlx installed successfully

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Updating local repository: mlx-lm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
...
```

### Without Local Repos

If no local repositories are found:

```text
[update.sh] Checking for local MLX development repositories...
[update.sh] No local MLX development repositories found
[update.sh] (Looked for mlx, mlx-lm, mlx-vlm directories with .git at /Users/jrp/Documents/AI/mlx)
```

The script continues normally and installs MLX packages from PyPI.

## Benefits

### For Development

1. **Automatic updates** - One command updates all MLX repos and reinstalls
2. **Correct order** - Installs in dependency order (mlx → mlx-lm → mlx-vlm)
3. **Editable installs** - Uses `pip install -e .` for live code changes
4. **Git integration** - Pulls latest changes from all repos

### For Testing

1. **Bleeding edge** - Test unreleased features immediately
2. **Custom patches** - Work with modified versions
3. **Version control** - Git manages all changes

### For Debugging

1. **Source access** - Full source code available for debugging
2. **Live updates** - Changes reflect immediately (no reinstall needed)
3. **Version tracking** - Git commits show exact versions in use

## Error Handling

### Git Pull Failures

If `git pull` fails (merge conflicts, network issues):

```text
⚠️  Git pull failed or had conflicts - skipping install for mlx
```

The script will:

- Skip installation for that repository
- Continue with remaining repositories
- Report the issue but not abort

### Installation Failures

If `pip install` fails:

```text
⚠️  Failed to install mlx-lm
```

The script will:

- Report the failure
- Continue with remaining repositories
- Complete other updates

## Technical Details

### Path Resolution

```bash
# Determine script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Calculate parent directory (3 levels up: tools → src → scripts → parent)
PARENT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
```

### Repository Detection

```bash
MLX_REPOS=("mlx" "mlx-lm" "mlx-vlm")

for repo in "${MLX_REPOS[@]}"; do
    REPO_PATH="$PARENT_DIR/$repo"
    if [[ -d "$REPO_PATH/.git" ]]; then
        # Process this repository
    fi
done
```

### Special Handling: mlx-vlm

```bash
if [[ "$repo" == "mlx-vlm" ]]; then
    echo "[update.sh] Installing opencv-python for mlx-vlm..."
    pip install -U opencv-python
fi
```

MLX-VLM requires OpenCV but doesn't always declare it properly in requirements.txt, so we install it explicitly.

## Integration with Existing Features

### Virtual Environment Check

The local repo update happens **after** the virtual environment check, so it's still protected from accidental global installations.

### SKIP_MLX Flag

If `SKIP_MLX=1` is set:

- Local repos are still updated via git pull
- But PyPI packages are also skipped
- Useful for testing without changing installed versions

### Force Reinstall

`FORCE_REINSTALL=1` does **not** affect local repo handling:

- Local repos always use `pip install -e .` (editable mode)
- Force reinstall only applies to PyPI packages

## Comparison: Before vs After

### Before This Change

```bash
# Manual process for updating local MLX builds
cd ~/mlx && git pull && pip install -e .
cd ~/mlx-lm && git pull && pip install -e .
cd ~/mlx-vlm && git pull && pip install -e .
cd ~/scripts/src/tools
SKIP_MLX=1 ./update.sh  # Skip PyPI versions
```

### After This Change

```bash
# Single command does everything
cd ~/scripts/src/tools
./update.sh
```

## Related Files

- `src/tools/update.sh` - Main update script
- `docs/notes/VIRTUAL_ENV_CHECK.md` - Virtual environment safety
- `docs/notes/UPDATE_TARGET_ADDED.md` - `make update` target documentation

## Testing

### Test Detection Without Running

```bash
cd /Users/jrp/Documents/AI/mlx/scripts/src/tools
bash -c '
SCRIPT_DIR="$(pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
for repo in mlx mlx-lm mlx-vlm; do
    [[ -d "$PARENT_DIR/$repo/.git" ]] && echo "✓ Found: $repo"
done
'
```

### Verify Editable Installs

After running `update.sh`:

```bash
pip show mlx | grep Location
pip show mlx-lm | grep Location
pip show mlx-vlm | grep Location
```

Look for paths pointing to local repositories (not site-packages).

## Future Enhancements

Potential improvements:

1. **Branch checking** - Warn if not on main/master branch
2. **Uncommitted changes** - Detect and warn about dirty repos
3. **Build verification** - Run quick smoke tests after install
4. **Parallel updates** - Update multiple repos concurrently
5. **Selective updates** - Flag to update only specific repos
