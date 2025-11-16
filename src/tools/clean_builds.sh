#!/usr/bin/env bash
# Clean build artifacts and caches from local MLX development repositories.
#
# This script removes:
# - Python build artifacts (build/, dist/, *.egg-info/, .eggs/)
# - Bytecode caches (__pycache__/, *.pyc, *.pyo)
# - Test caches (.pytest_cache/)
# - Type checking caches (.mypy_cache/)
#
# Note: This does NOT remove compiled Metal kernels from system caches.
# Metal kernel caches are managed by the Metal framework and cleared on reboot
# or via system cache cleaning utilities.
#
# Usage:
#   bash tools/clean_builds.sh              # Clean local MLX repos + scripts project
#   bash tools/clean_builds.sh --dry-run    # Show what would be cleaned
#   bash tools/clean_builds.sh --help       # Show this help

set -euo pipefail

DRY_RUN=0

# Parse arguments
for arg in "$@"; do
	case $arg in
		--dry-run)
			DRY_RUN=1
			shift
			;;
		--help|-h)
			head -n 17 "$0" | tail -n +2 | sed 's/^# //'
			exit 0
			;;
		*)
			echo "Unknown option: $arg"
			echo "Run with --help for usage information"
			exit 1
			;;
	esac
done

# Determine directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Repositories to clean (in order)
MLX_REPOS=("mlx" "mlx-lm" "mlx-vlm" "mlx-data")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧹 MLX Build Artifact Cleanup"
if [[ $DRY_RUN -eq 1 ]]; then
	echo "   DRY RUN MODE - No files will be deleted"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TOTAL_CLEANED=0

# Clean function
clean_directory() {
	local dir=$1
	local name=$2
	
	if [[ ! -d "$dir" ]]; then
		return
	fi
	
	echo "Cleaning: $name"
	echo "  Path: $dir"
	
	local cleaned=0
	
	# Build directories
	for pattern in "build" "dist" ".eggs"; do
		if [[ -d "$dir/$pattern" ]]; then
			if [[ $DRY_RUN -eq 1 ]]; then
				echo "  [DRY RUN] Would remove: $pattern/"
			else
				rm -rf "${dir:?}/$pattern"
				echo "  ✓ Removed: $pattern/"
			fi
			((cleaned++))
		fi
	done
	
	# Egg-info directories
	while IFS= read -r -d '' egginfo; do
		if [[ $DRY_RUN -eq 1 ]]; then
			echo "  [DRY RUN] Would remove: $(basename "$egginfo")"
		else
			rm -rf "$egginfo"
			echo "  ✓ Removed: $(basename "$egginfo")"
		fi
		((cleaned++))
	done < <(find "$dir" -maxdepth 1 -type d -name '*.egg-info' -print0 2>/dev/null)
	
	# Cache directories
	for cache in "__pycache__" ".pytest_cache" ".mypy_cache"; do
		local count=0
		if [[ $DRY_RUN -eq 1 ]]; then
			count=$(find "$dir" -type d -name "$cache" 2>/dev/null | wc -l | tr -d ' ')
			if [[ $count -gt 0 ]]; then
				echo "  [DRY RUN] Would remove: $count $cache directories"
				((cleaned++))
			fi
		else
			while IFS= read -r -d '' cachedir; do
				rm -rf "$cachedir"
				((count++))
			done < <(find "$dir" -type d -name "$cache" -print0 2>/dev/null)
			if [[ $count -gt 0 ]]; then
				echo "  ✓ Removed: $count $cache directories"
				((cleaned++))
			fi
		fi
	done
	
	# Bytecode files
	local pyc_count=0
	if [[ $DRY_RUN -eq 1 ]]; then
		pyc_count=$(find "$dir" -type f \( -name '*.pyc' -o -name '*.pyo' \) 2>/dev/null | wc -l | tr -d ' ')
		if [[ $pyc_count -gt 0 ]]; then
			echo "  [DRY RUN] Would remove: $pyc_count bytecode files"
			((cleaned++))
		fi
	else
		while IFS= read -r -d '' pycfile; do
			rm -f "$pycfile"
			((pyc_count++))
		done < <(find "$dir" -type f \( -name '*.pyc' -o -name '*.pyo' \) -print0 2>/dev/null)
		if [[ $pyc_count -gt 0 ]]; then
			echo "  ✓ Removed: $pyc_count bytecode files"
			((cleaned++))
		fi
	fi
	
	if [[ $cleaned -eq 0 ]]; then
		echo "  (no artifacts found)"
	fi
	
	TOTAL_CLEANED=$((TOTAL_CLEANED + cleaned))
	echo ""
}

# Clean local MLX repositories
for repo in "${MLX_REPOS[@]}"; do
	REPO_PATH="$PARENT_DIR/$repo"
	if [[ -d "$REPO_PATH/.git" ]]; then
		clean_directory "$REPO_PATH" "$repo (local dev)"
	fi
done

# Clean scripts project
clean_directory "$SCRIPTS_DIR/src" "mlx-vlm-check (scripts)"

# Clean typings
if [[ -d "$SCRIPTS_DIR/typings" ]]; then
	echo "Cleaning: Type stubs"
	echo "  Path: $SCRIPTS_DIR/typings"
	if [[ $DRY_RUN -eq 1 ]]; then
		echo "  [DRY RUN] Would remove: typings/"
	else
		rm -rf "$SCRIPTS_DIR/typings"
		echo "  ✓ Removed: typings/"
	fi
	((TOTAL_CLEANED++))
	echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ $DRY_RUN -eq 1 ]]; then
	echo "DRY RUN COMPLETE"
	echo "Run without --dry-run to actually remove files"
else
	if [[ $TOTAL_CLEANED -gt 0 ]]; then
		echo "✓ Cleanup complete - removed $TOTAL_CLEANED artifact groups"
	else
		echo "✓ All clean - no artifacts found"
	fi
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Note: Metal kernel caches are managed by macOS and persist in"
echo "system temp directories. They're automatically cleared on reboot."
echo "To manually clear: sudo rm -rf /tmp/com.apple.metal/* (use with caution)"
echo ""
echo "Build artifacts cleaned. To rebuild MLX repos from scratch:"
echo "  cd /path/to/mlx && cmake -S . -B build && cmake --build build"
echo "  Then run: pip install -e ."
