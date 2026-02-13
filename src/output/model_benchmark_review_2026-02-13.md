# Model Benchmark Results Review — Actionable Summary

This document summarizes the results of the latest `check_models` run (2026-02-13), categorizing models by failure mode and output quality. Use this as a reference for triage, debugging, and improvement.

---

## :x: Models That Crash or Fail to Run

These models failed to produce any output due to errors such as weight mismatches, missing chat templates, out-of-memory, or tokenizer/processor issues:

- `microsoft/Florence-2-large-ft` (Weight Mismatch)
- `mlx-community/GLM-4.6V-Flash-6bit` (Model Error: shape broadcast)
- `mlx-community/GLM-4.6V-Flash-mxfp4` (Model Error: shape broadcast)
- `mlx-community/Idefics3-8B-Llama3-bf16` (Model Error: weight shape mismatch)
- `mlx-community/Kimi-VL-A3B-Thinking-8bit` (Model Error: shape broadcast)
- `mlx-community/X-Reasoner-7B-8bit` (OOM: attempted to allocate >128GB)
- `mlx-community/gemma-3n-E2B-4bit` (No Chat Template)
- `mlx-community/paligemma2-3b-pt-896-4bit` (Model Error: shape broadcast)
- `prince-canuma/Florence-2-large-ft` (Model Error: tokenizer attribute missing)

**Action:** Investigate model weights, configuration, and compatibility with the current MLX/MLX-VLM stack. See error details in `results.md` for stack traces.

---

## :warning: Models Producing Gibberish, Repetitive, or Corrupted Output

These models produced output that is repetitive, nonsensical, or shows clear signs of misconfiguration (e.g., repeated phrases, hallucinated content):

- `Qwen/Qwen3-VL-2B-Instruct` (Repetitive: "churchyard, churchyard, ...")
- `microsoft/Phi-3.5-vision-instruct` (Repetitive: "weather, weather, ...")
- `mlx-community/Phi-3.5-vision-instruct-bf16` (Repetitive: "weather, weather, ...")
- `mlx-community/Qwen2-VL-2B-Instruct-4bit` (Repetitive: "travel photography, ...")
- `mlx-community/deepseek-vl2-8bit` (Repetitive: "English country church, ..." repeated)
- `mlx-community/paligemma2-3b-ft-docci-448-bf16` (Repetitive: "the windows are set..." repeated)

**Action:** Check tokenizer/model alignment, prompt formatting, and ensure correct model variant is loaded.

---

## :triangular_flag_on_post: Models With Low Quality Output

These models ran but produced low utility scores (D/F), often echoing context, being too short, or missing key details:

- `Qwen/Qwen3-VL-2B-Instruct` (F, 27/100) — echoes context
- `microsoft/Phi-3.5-vision-instruct` (F, 26/100) — echoes context
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (F, 7/100) — output too short
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (D, 43/100) — echoes context
- `mlx-community/InternVL3-14B-8bit` (F, 31/100) — echoes context
- `mlx-community/LFM2-VL-1.6B-8bit` (F, 26/100) — echoes context
- `mlx-community/LFM2.5-VL-1.6B-bf16` (F, 24/100) — echoes context
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (D, 47/100) — echoes context
- `mlx-community/Molmo-7B-D-0924-8bit` (F, 0/100) — empty/minimal output
- `mlx-community/Molmo-7B-D-0924-bf16` (F, 0/100) — empty/minimal output
- `mlx-community/Phi-3.5-vision-instruct-bf16` (F, 26/100) — echoes context
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (F, 21/100) — echoes context
- `mlx-community/deepseek-vl2-8bit` (D, 45/100) — echoes context
- `mlx-community/gemma-3-27b-it-qat-8bit` (D, 47/100) — echoes context
- `mlx-community/llava-v1.6-mistral-7b-8bit` (F, 25/100) — echoes context
- `mlx-community/paligemma2-10b-ft-docci-448-6bit` (F, 5/100) — too short
- `mlx-community/paligemma2-10b-ft-docci-448-bf16` (F, 23/100) — lacks visual description
- `mlx-community/paligemma2-3b-ft-docci-448-bf16` (D, 48/100) — lacks visual description
- `mlx-community/pixtral-12b-8bit` (F, 24/100) — echoes context
- `mlx-community/pixtral-12b-bf16` (D, 39/100) — echoes context
- `qnguyen3/nanoLLaVA` (F, 15/100) — echoes context

**Action:** Review prompt handling, model fine-tuning, and evaluation criteria. Consider retraining or prompt engineering.

---

## :star: High Quality Models

These models produced accurate, relevant, and well-structured output, passing all quality checks:

- `mlx-community/gemma-3-27b-it-qat-4bit` (A, 88/100) — **Best for cataloging**
- `mlx-community/LFM2-VL-1.6B-8bit`
- `mlx-community/LFM2.5-VL-1.6B-bf16`
- `mlx-community/SmolVLM-Instruct-bf16`
- `HuggingFaceTB/SmolVLM-Instruct`
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`
- `mlx-community/llava-v1.6-mistral-7b-8bit`
- `mlx-community/pixtral-12b-bf16`
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`
- `mlx-community/gemma-3-27b-it-qat-8bit`
- `mlx-community/InternVL3-14B-8bit`
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`
- `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`
- `mlx-community/gemma-3n-E4B-it-bf16`
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`
- `Qwen/Qwen3-VL-2B-Instruct`
- `microsoft/Phi-3.5-vision-instruct`
- `mlx-community/Phi-3.5-vision-instruct-bf16`
- `mlx-community/deepseek-vl2-8bit`
- `mlx-community/Molmo-7B-D-0924-8bit`
- `mlx-community/Molmo-7B-D-0924-bf16`
- `qnguyen3/nanoLLaVA`

**Action:** No immediate action required. Use these models as reference for others.

---

## Next Steps

- For each model in the above categories, please:
  1. Attach relevant log excerpts from `src/output/results.md` as needed.
  2. Assign owners for investigation and remediation.
  3. Update this document with findings and resolutions.

---

**Note:** For full details, see `src/output/results.md`.
