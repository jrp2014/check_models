# MLX Documentation Consistency Check

**Date**: 2025-10-12  
**Purpose**: Verify our local MLX setup aligns with official documentation

## Official Documentation Sources

- **MLX Core**: <https://ml-explore.github.io/mlx/build/html/install.html>
- **MLX-LM**: <https://github.com/ml-explore/mlx-lm>
- **MLX-VLM**: <https://github.com/Blaizzy/mlx-vlm> (community project, not ml-explore)

## Findings

### ✅ MLX Core - CONSISTENT

Our implementation matches the official documentation:

#### Official Build Instructions (from ml-explore docs)

```bash
# Clone repository
git clone git@github.com:ml-explore/mlx.git mlx && cd mlx

# Standard install
pip install .

# Development install
pip install -e ".[dev]"

# Fast build for development
python setup.py build_ext --inplace

# Generate stubs
python setup.py generate_stubs
```

#### Our Implementation

```bash
# In update.sh
pip install -e .  # ✓ Correct
python setup.py generate_stubs  # ✓ Correct
```

**Status**: ✅ **PERFECT MATCH** - Our script correctly uses the official methods.

### ✅ MLX-LM - CONSISTENT

#### Official Instructions

```bash
# From PyPI
pip install mlx-lm

# Or with conda
conda install -c conda-forge mlx-lm
```

**No specific build-from-source instructions** in the README, just standard `pip install .`

#### Our Implementation

```bash
pip install -e .  # ✓ Standard editable install
```

**Status**: ✅ **CONSISTENT** - Uses standard Python packaging, no special steps needed.

### ⚠️ MLX-VLM - ATTENTION NEEDED

#### Official Instructions

```bash
# From PyPI
pip install -U mlx-vlm
```

**Repository**: <https://github.com/Blaizzy/mlx-vlm> (community project)

#### Key Details

1. **requirements.txt exists** - Contains `opencv-python>=4.12.0.88`
2. **No special build instructions** - Standard pip install
3. **No stub generation** - Not mentioned in documentation

#### Our Implementation

```bash
# Special handling for mlx-vlm
pip install -U -r requirements.txt  # Installs opencv-python>=4.12.0.88
pip install -U opencv-python  # ⚠️ Redundant - already in requirements.txt
pip install -e .
```

**Issues Identified**:

1. ⚠️ Redundant opencv-python installation (already in requirements.txt)
2. ⚠️ Order issue: We install opencv-python AFTER requirements.txt (redundant upgrade)
3. ℹ️ Not harmful, just unnecessary

### 📋 Recommended Changes

#### 1. Fix OpenCV Installation for mlx-vlm

**Current code (incorrect)**:

```bash
# Install requirements.txt if it exists
if [[ -f "requirements.txt" ]]; then
    echo "[update.sh] Installing requirements from requirements.txt..."
    pip install -U -r requirements.txt
else
    echo "[update.sh] No requirements.txt found"
fi

# Special handling for mlx-vlm: install opencv-python
if [[ "$repo" == "mlx-vlm" ]]; then
    echo "[update.sh] Installing opencv-python for mlx-vlm..."
    pip install -U opencv-python
fi
```

**Recommended fix** (simpler solution):

```bash
# Remove redundant special opencv-python handling
# Let requirements.txt handle it (it includes opencv-python>=4.12.0.88)
if [[ -f "requirements.txt" ]]; then
    echo "[update.sh] Installing requirements from requirements.txt..."
    pip install -U -r requirements.txt
else
    echo "[update.sh] No requirements.txt found"
fi

# No special opencv-python handling needed for mlx-vlm
```

**Rationale**: The mlx-vlm requirements.txt already includes opencv-python>=4.12.0.88, so our explicit installation is redundant. The original comment mentioned "particular numpy dependency requirements" but this is handled by requirements.txt.

#### 2. Update Documentation Comment

**Current**:

```bash
# Special handling for mlx-vlm: install opencv-python (which has particular numpy dependency reqruirements)
```

**Should be**:

```bash
# Note: mlx-vlm requirements.txt includes opencv-python-headless
# If you need the full opencv-python (with GUI support), install it separately
```

## Summary

### What's Working Correctly

1. ✅ MLX core build and stub generation
2. ✅ MLX-LM standard installation
3. ✅ Overall structure and order of operations
4. ✅ Virtual environment checks
5. ✅ Git pull before installation
6. ✅ Editable installs (`pip install -e .`)

### What Needs Attention

1. ⚠️ **Redundant opencv-python installation** - mlx-vlm requirements.txt already includes it
2. ℹ️ The special handling was added based on outdated information or a misunderstanding

### Recommendation

**Remove the redundant opencv-python handling** (Simple fix)

- The mlx-vlm requirements.txt already specifies `opencv-python>=4.12.0.88`
- No special handling needed
- Simplifies the code
- Matches the official mlx-vlm setup

**Implementation**: Simply remove the special opencv-python block entirely.

## Additional Notes

### MLX Core Build Requirements

From official docs:

- macOS >= 13.5 (preferably macOS 14+)
- Apple Silicon (M series chips)
- Python >= 3.9 (we use 3.13 ✓)
- Xcode >= 15.0 with macOS SDK >= 14.0
- cmake >= 3.25 (we specify 3.25,<4.1 ✓)

### MLX-VLM is a Community Project

- **Not** an official ml-explore project
- Maintained by @Blaizzy
- Different from mlx-vision (which is a 404)
- May have different conventions

## Testing Recommendation

```bash
# Test the current setup
cd /Users/jrp/Documents/AI/mlx/mlx-vlm
pip list | grep opencv

# Should show either:
# opencv-python       or
# opencv-python-headless
# (but not both to avoid conflicts)
```
