# Virtual Environment Check Added to update.sh

**Date**: 2025-10-12  
**Context**: Safety improvement for dependency management

## Change Summary

Added a safety check to `src/tools/update.sh` to warn users if they are not running in a virtual environment (conda, venv, or uv).

## Problem

The `update.sh` script updates Python packages using pip. If run outside a virtual environment, it would install packages globally on the system, which can:

- Pollute the system Python installation
- Cause version conflicts with system packages
- Create hard-to-debug environment issues
- Require sudo/admin privileges in some cases

## Solution

Added environment detection that checks for:

- `CONDA_DEFAULT_ENV` - conda environments
- `VIRTUAL_ENV` - venv/virtualenv environments  
- `UV_ACTIVE` - uv virtual environments

If none of these variables are set, the script:

1. Displays a clear warning message
2. Explains the risks
3. Shows how to activate each type of environment
4. Prompts for user confirmation before proceeding
5. Aborts if user chooses not to continue

## Implementation

```bash
# Check if we're in a virtual environment (uv, conda, venv, virtualenv)
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -z "${CONDA_DEFAULT_ENV:-}" ]] && [[ -z "${UV_ACTIVE:-}" ]]; then
    echo "⚠️  WARNING: You don't appear to be in a virtual environment!"
    echo "   (No VIRTUAL_ENV, CONDA_DEFAULT_ENV, or UV_ACTIVE detected)"
    echo ""
    echo "   This script will update packages globally on your system."
    echo "   It's strongly recommended to activate a virtual environment first:"
    echo ""
    echo "   • conda: conda activate <env-name>"
    echo "   • venv/virtualenv: source /path/to/venv/bin/activate"
    echo "   • uv: uv venv && source .venv/bin/activate"
    echo ""
    read -p "   Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "[update.sh] Aborted by user."
        exit 1
    fi
    echo "[update.sh] Proceeding with global installation (user confirmed)..."
fi
```

## Benefits

1. **Prevents accidental global installations** - Catches the most common mistake
2. **Educational** - Shows users how to activate different environment types
3. **Non-blocking** - Advanced users can still proceed if needed
4. **Clear feedback** - Explicit about what's happening and why

## Related Scripts

- `validate_env.py` - Already has conda environment checks
- `setup_conda_env.sh` - Creates conda environments (doesn't need this check)
- `update.sh` - Now has the safety check ✓

## Testing

Tested with:

- Active conda environment (mlx-vlm) - passes silently ✓
- No virtual environment - shows warning and prompts ✓
- Environment variables unset - correctly detects and warns ✓
