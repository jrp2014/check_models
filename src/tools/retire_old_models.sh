#!/usr/bin/env bash
# Retire superseded check_models retirement candidates from the Hugging Face cache.
#
# A candidate is "old" when a newer version of the same model line exists
# (assessment of 2026-10-23, against the candidates listed in
# https://github.com/Blaizzy/mlx-vlm/issues/2546). Two tiers:
#   - default: the successor is already in the roster (43.2 GB)
#   - --upstream: the successor exists only upstream (a further 179.9 GB)
# Candidates with no newer version (nanoLLaVA-1.5, FastVLM-0.5B,
# InternVL3.5-1B, ERNIE-4.5-VL, X-Reasoner-7B, MolmoPoint-8B, diffusiongemma,
# llm-jp-4-vl, Muse-Glimmer, Nemotron-3.5-Bolt) are deliberately left alone.
#
# Deletion goes through `hf cache rm`, never `rm -rf`: Hub 1.32+ keeps
# weights in a hub-wide blob store shared between repos, and `hf cache rm`
# frees the shared blobs only the removed repos used. --apply deletes exactly
# what the dry run lists. It never runs `hf cache prune`, which also removes
# detached revisions across the whole cache (such as snapshots kept for
# reproduction); preview that separately with `hf cache prune --dry-run`.
#
# Usage: bash src/tools/retire_old_models.sh [--upstream] [--apply]
#   without --apply it is a dry run that prints what would go and its size.
set -euo pipefail

if [[ -n "${HF:-}" ]]; then
  hf_bin="$HF"
elif command -v hf >/dev/null 2>&1; then
  hf_bin="$(command -v hf)"
else
  hf_bin="$HOME/miniconda3/envs/mlx-vlm/bin/hf"
fi

apply=0
upstream=0
for arg in "$@"; do
  case "$arg" in
    --apply) apply=1 ;;
    --upstream) upstream=1 ;;
    -h | --help)
      sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown option: $arg (see --help)" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$hf_bin" ]]; then
  echo "Hugging Face CLI not found at $hf_bin; set HF=/path/to/hf." >&2
  exit 1
fi

# Removing a model while a sweep reads the cache would fail that model mid-run.
if pgrep -f 'python.*-m check_models' >/dev/null 2>&1; then
  echo "A check_models sweep is running; try again when it finishes." >&2
  exit 1
fi

# Successor already in the roster.
targets=(
  model/mlx-community/Qwen2-VL-2B-mlx             # -> Qwen3-VL / Qwen3.5 / Qwen3.8
  model/mlx-community/gemma-3-27b-it-qat-4bit     # -> Gemma 4
  model/mlx-community/gemma-3n-E4B-it-4bit        # -> gemma-4-e4b; usable 2026-10-23, the first sweep with mlx-vlm#2548
  model/mlx-community/SmolVLM-256M-Instruct-4bit  # -> SmolVLM2
  model/mlx-community/granite-vision-3.2-2b-nvfp4 # -> granite-4.0-3b-vision
  model/mlx-community/pixtral-12b-8bit            # -> Mistral-Small-3.2 / Ministral-3
)

# Successor exists only upstream.
if ((upstream)); then
  targets+=(
    model/vikhyatk/moondream2                              # -> moondream3
    model/moondream/starmie-v1                             # moondream2's tokenizer; goes with it
    model/mlx-community/Llama-3.2-11B-Vision-Instruct-8bit # -> Llama 4
    model/mlx-community/Kimi-VL-A3B-Thinking-2506-8bit     # -> Kimi K2.5-K3
    model/mlx-community/GLM-4.6V-nvfp4                     # -> GLM-5.3-Flash
    model/mlx-community/Step-3.7-Flash-oQ3e                # -> Step-5-Preview
  )
fi

if ((apply)); then
  "$hf_bin" cache rm "${targets[@]}" --yes
else
  "$hf_bin" cache rm "${targets[@]}" --dry-run
  echo "Dry run only: nothing was deleted. Re-run with --apply to delete."
fi
