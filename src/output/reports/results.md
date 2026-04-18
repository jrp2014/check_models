# Model Performance Results

_Generated on 2026-04-19 00:43:13 BST_

## 🎯 Action Snapshot

**Failures & Triage**

- _Framework/runtime failures:_ 53 (top owners: huggingface-hub=53).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=0, clean outputs=0/1.
- _Useful now:_ none (no clean A/B shortlist for this run).
- _Review watchlist:_ 1 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.

**Quality & Metadata**

- _Vs existing metadata:_ better=0, neutral=0, worse=1 (baseline B 75/100).
- _Quality signal frequency:_ cutoff=1, trusted_hint_ignored=1,
  context_ignored=1, repetitive=1, missing_sections=1.
- _Termination reasons:_ completed=1, exception=53.

**Runtime**

- _Runtime pattern:_ decode dominates measured phase time (52%; 1/54 measured
  model(s)).
- _Phase totals:_ model load=9.29s, prompt prep=0.00s, decode=15.88s,
  cleanup=5.11s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=1, exception=53.

## 🏆 Performance Highlights

- **Fastest:** `HuggingFaceTB/SmolVLM-Instruct` (49.1 tps)
- **💾 Most efficient:** `HuggingFaceTB/SmolVLM-Instruct` (5.5 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.69s)
- **📊 Average TPS:** 49.1 across 1 models

## 📈 Resource Usage

- **Total peak memory:** 5.5 GB
- **Average peak memory:** 5.5 GB
- **Memory efficiency:** 413 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🟠 D: 1

**Average Utility Score:** 45/100

**Existing Metadata Baseline:** ✅ B (75/100)
**Vs Existing Metadata:** Avg Δ -30 | Better: 0, Neutral: 0, Worse: 1

- **Best for cataloging:** `HuggingFaceTB/SmolVLM-Instruct` (🟠 D, 45/100)
- **Best descriptions:** `HuggingFaceTB/SmolVLM-Instruct` (60/100)
- **Best keywording:** `HuggingFaceTB/SmolVLM-Instruct` (0/100)
- **Worst for cataloging:** `HuggingFaceTB/SmolVLM-Instruct` (🟠 D, 45/100)

### ⚠️ 1 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (53):**
  - `Qwen/Qwen3-VL-2B-Instruct` (`Model Error`)
  - `ggml-org/gemma-3-1b-it-GGUF` (`Model Error`)
  - `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (`Model Error`)
  - `meta-llama/Llama-3.2-11B-Vision-Instruct` (`Model Error`)
  - `microsoft/Phi-3.5-vision-instruct` (`Model Error`)
  - `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (`Model Error`)
  - `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (`Model Error`)
  - `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (`Model Error`)
  - `mlx-community/FastVLM-0.5B-bf16` (`Model Error`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (`Model Error`)
  - `mlx-community/GLM-4.6V-Flash-mxfp4` (`Model Error`)
  - `mlx-community/GLM-4.6V-nvfp4` (`Model Error`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (`Model Error`)
  - `mlx-community/InternVL3-14B-8bit` (`Model Error`)
  - `mlx-community/InternVL3-8B-bf16` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2-VL-1.6B-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Model Error`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (`Model Error`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (`Model Error`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` (`Model Error`)
  - `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (`Model Error`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (`Model Error`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Model Error`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (`Model Error`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (`Model Error`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (`Model Error`)
  - `mlx-community/Qwen3.5-27B-4bit` (`Model Error`)
  - `mlx-community/Qwen3.5-27B-mxfp8` (`Model Error`)
  - `mlx-community/Qwen3.5-35B-A3B-4bit` (`Model Error`)
  - `mlx-community/Qwen3.5-35B-A3B-6bit` (`Model Error`)
  - `mlx-community/Qwen3.5-35B-A3B-bf16` (`Model Error`)
  - `mlx-community/Qwen3.5-9B-MLX-4bit` (`Model Error`)
  - `mlx-community/SmolVLM-Instruct-bf16` (`Model Error`)
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (`Model Error`)
  - `mlx-community/X-Reasoner-7B-8bit` (`Model Error`)
  - `mlx-community/gemma-3-27b-it-qat-4bit` (`Model Error`)
  - `mlx-community/gemma-3-27b-it-qat-8bit` (`Model Error`)
  - `mlx-community/gemma-3n-E2B-4bit` (`Model Error`)
  - `mlx-community/gemma-3n-E4B-it-bf16` (`Model Error`)
  - `mlx-community/gemma-4-31b-bf16` (`Model Error`)
  - `mlx-community/gemma-4-31b-it-4bit` (`Model Error`)
  - `mlx-community/llava-v1.6-mistral-7b-8bit` (`Model Error`)
  - `mlx-community/nanoLLaVA-1.5-4bit` (`Model Error`)
  - `mlx-community/paligemma2-10b-ft-docci-448-6bit` (`Model Error`)
  - `mlx-community/paligemma2-10b-ft-docci-448-bf16` (`Model Error`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (`Model Error`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (`Model Error`)
  - `mlx-community/pixtral-12b-8bit` (`Model Error`)
  - `mlx-community/pixtral-12b-bf16` (`Model Error`)
  - `qnguyen3/nanoLLaVA` (`Model Error`)
- **🔄 Repetitive Output (1):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 49.1 | Min: 49.1 | Max: 49.1
- **Peak Memory**: Avg: 5.5 | Min: 5.5 | Max: 5.5
- **Total Time**: Avg: 17.05s | Min: 17.05s | Max: 17.05s
- **Generation Time**: Avg: 15.88s | Min: 15.88s | Max: 15.88s
- **Model Load Time**: Avg: 0.69s | Min: 0.69s | Max: 0.69s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (52%; 1/54 measured model(s)).
- **Phase totals:** model load=9.29s, prompt prep=0.00s, decode=15.88s, cleanup=5.11s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=1, exception=53.

### ⏱ Timing Snapshot

- **Validation overhead:** 19.77s total (avg 0.37s across 54 model(s)).
- **First-token latency:** Avg 4.83s | Min 4.83s | Max 4.83s across 1 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct)
  (D 45/100 | Desc 60 | Keywords 0 | Gen 49.1 TPS | Peak 5.5 | D 45/100 | hit
  token cap (500) | nontext prompt burden=73% | missing sections: title,
  description, keywords | missing terms: 10 Best, 10 Best (structured), Bay
  Window, Berkshire, Blue sky)
- _Best descriptions:_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct)
  (D 45/100 | Desc 60 | Keywords 0 | Gen 49.1 TPS | Peak 5.5 | D 45/100 | hit
  token cap (500) | nontext prompt burden=73% | missing sections: title,
  description, keywords | missing terms: 10 Best, 10 Best (structured), Bay
  Window, Berkshire, Blue sky)
- _Best keywording:_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct)
  (D 45/100 | Desc 60 | Keywords 0 | Gen 49.1 TPS | Peak 5.5 | D 45/100 | hit
  token cap (500) | nontext prompt burden=73% | missing sections: title,
  description, keywords | missing terms: 10 Best, 10 Best (structured), Bay
  Window, Berkshire, Blue sky)
- _Fastest generation:_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct)
  (D 45/100 | Desc 60 | Keywords 0 | Gen 49.1 TPS | Peak 5.5 | D 45/100 | hit
  token cap (500) | nontext prompt burden=73% | missing sections: title,
  description, keywords | missing terms: 10 Best, 10 Best (structured), Bay
  Window, Berkshire, Blue sky)
- _Lowest memory footprint:_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct)
  (D 45/100 | Desc 60 | Keywords 0 | Gen 49.1 TPS | Peak 5.5 | D 45/100 | hit
  token cap (500) | nontext prompt burden=73% | missing sections: title,
  description, keywords | missing terms: 10 Best, 10 Best (structured), Bay
  Window, Berkshire, Blue sky)
- _Best balance:_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct)
  (D 45/100 | Desc 60 | Keywords 0 | Gen 49.1 TPS | Peak 5.5 | D 45/100 | hit
  token cap (500) | nontext prompt burden=73% | missing sections: title,
  description, keywords | missing terms: 10 Best, 10 Best (structured), Bay
  Window, Berkshire, Blue sky)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (53):_ [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`ggml-org/gemma-3-1b-it-GGUF`](model_gallery.md#model-ggml-org-gemma-3-1b-it-gguf),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  +49 more. Example: `Model Error`.
- _🔄 Repetitive Output (1):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct).
  Example: token: `unt`.
- _Low-utility outputs (1):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct).
  Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `huggingface-hub` | 53 | Model Error | `Qwen/Qwen3-VL-2B-Instruct`, `ggml-org/gemma-3-1b-it-GGUF`, `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`, `meta-llama/Llama-3.2-11B-Vision-Instruct`, `microsoft/Phi-3.5-vision-instruct`, `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`, `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/GLM-4.6V-Flash-6bit`, `mlx-community/GLM-4.6V-Flash-mxfp4`, `mlx-community/GLM-4.6V-nvfp4`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-14B-8bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Molmo-7B-D-0924-8bit`, `mlx-community/Molmo-7B-D-0924-bf16`, `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3.5-27B-4bit`, `mlx-community/Qwen3.5-27B-mxfp8`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-35B-A3B-6bit`, `mlx-community/Qwen3.5-35B-A3B-bf16`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/SmolVLM-Instruct-bf16`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-3-27b-it-qat-8bit`, `mlx-community/gemma-3n-E2B-4bit`, `mlx-community/gemma-3n-E4B-it-bf16`, `mlx-community/gemma-4-31b-bf16`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/llava-v1.6-mistral-7b-8bit`, `mlx-community/nanoLLaVA-1.5-4bit`, `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`, `mlx-community/pixtral-12b-8bit`, `mlx-community/pixtral-12b-bf16`, `qnguyen3/nanoLLaVA` |

### Actionable Items by Package

#### huggingface-hub

- Qwen/Qwen3-VL-2B-Instruct (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- ggml-org/gemma-3-1b-it-GGUF (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- jqlive/Kimi-VL-A3B-Thinking-2506-6bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- meta-llama/Llama-3.2-11B-Vision-Instruct (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- microsoft/Phi-3.5-vision-instruct (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/FastVLM-0.5B-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/GLM-4.6V-Flash-6bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/GLM-4.6V-Flash-mxfp4 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/GLM-4.6V-nvfp4 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Idefics3-8B-Llama3-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/InternVL3-14B-8bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/InternVL3-8B-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/LFM2-VL-1.6B-8bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/LFM2.5-VL-1.6B-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Llama-3.2-11B-Vision-Instruct-8bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Ministral-3-3B-Instruct-2512-4bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Molmo-7B-D-0924-8bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Molmo-7B-D-0924-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/MolmoPoint-8B-fp16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Phi-3.5-vision-instruct-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Qwen2-VL-2B-Instruct-4bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Qwen3-VL-2B-Thinking-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Qwen3.5-27B-4bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Qwen3.5-27B-mxfp8 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Qwen3.5-35B-A3B-4bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Qwen3.5-35B-A3B-6bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Qwen3.5-35B-A3B-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/Qwen3.5-9B-MLX-4bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/SmolVLM-Instruct-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/SmolVLM2-2.2B-Instruct-mlx (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/X-Reasoner-7B-8bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/gemma-3-27b-it-qat-4bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/gemma-3-27b-it-qat-8bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/gemma-3n-E2B-4bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/gemma-3n-E4B-it-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/gemma-4-31b-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/gemma-4-31b-it-4bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/llava-v1.6-mistral-7b-8bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/nanoLLaVA-1.5-4bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/paligemma2-10b-ft-docci-448-6bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/paligemma2-10b-ft-docci-448-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/paligemma2-3b-ft-docci-448-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/paligemma2-3b-pt-896-4bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/pixtral-12b-8bit (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- mlx-community/pixtral-12b-bf16 (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`
- qnguyen3/nanoLLaVA (Model Error)
  - Error: `Model loading failed: [Errno 32] Broken pipe`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD028 MD037 -->
>
> Analyze this image for cataloguing metadata, using British English.
>
> Use only details that are clearly and definitely visible in the image. If a
> detail is uncertain, ambiguous, partially obscured, too small to verify, or
> not directly visible, leave it out. Do not guess.
>
> Treat the metadata hints below as a draft catalog record. Keep only details
> that are clearly confirmed by the image, correct anything contradicted by
> the image, and add important visible details that are definitely present.
>
> &#8203;Return exactly these three sections, and nothing else:
>
> &#8203;Title:
> &#45; 5-10 words, concrete and factual, limited to clearly visible content.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> &#8203;Description:
> &#45; 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> &#45; Output only the description text after the label.
>
> &#8203;Keywords:
> &#45; 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> &#45; Output only the keyword list after the label.
>
> &#8203;Rules:
> &#45; Include only details that are definitely visible in the image.
> &#45; Reuse metadata terms only when they are clearly supported by the image.
> &#45; If metadata and image disagree, follow the image.
> &#45; Prefer omission to speculation.
> &#45; Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent unless visually obvious.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Existing metadata hints (high confidence; use only when visually
> &#8203;confirmed):
> &#45; Description hint: The late afternoon sun illuminates the Tudor Revival
> architecture of the buildings at Lincoln's Inn in London, England. The
> historic red brick structures, with their distinctive diaper patterning,
> stone quoins, and large bay windows, are part of one of the four Inns of
> Court in London to which barristers of England and Wales belong.
> &#45; Keyword hints: 10 Best, 10 Best (structured), Adobe Stock, Any Vision, Bay
> Window, Berkshire, Blue sky, Bollard, Brickwork, Car, Chimney, Chimneys,
> Daylight, Door, Driving, England, Eton, Eton College, Europe, Gable
> &#45; Capture metadata: Taken on 2026-04-18 18:10:10 BST (at 18:10:10 local
> time). GPS: 52.203500°N, 0.113500°E.
<!-- markdownlint-enable MD028 MD037 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 50.92s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues       |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:---------------------|----------------:|
| `Qwen/Qwen3-VL-2B-Instruct`                             |         |                   |                       |                |              |           |             |                  |      0.13s |       0.60s |                      | huggingface-hub |
| `ggml-org/gemma-3-1b-it-GGUF`                           |         |                   |                       |                |              |           |             |                  |      0.24s |       0.72s |                      | huggingface-hub |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |         |                   |                       |                |              |           |             |                  |      0.27s |       0.73s |                      | huggingface-hub |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |         |                   |                       |                |              |           |             |                  |      0.19s |       0.63s |                      | huggingface-hub |
| `microsoft/Phi-3.5-vision-instruct`                     |         |                   |                       |                |              |           |             |                  |      0.21s |       0.64s |                      | huggingface-hub |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |         |                   |                       |                |              |           |             |                  |      0.11s |       0.46s |                      | huggingface-hub |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |         |                   |                       |                |              |           |             |                  |      0.10s |       0.40s |                      | huggingface-hub |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |         |                   |                       |                |              |           |             |                  |      0.11s |       0.40s |                      | huggingface-hub |
| `mlx-community/FastVLM-0.5B-bf16`                       |         |                   |                       |                |              |           |             |                  |      0.12s |       0.42s |                      | huggingface-hub |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |         |                   |                       |                |              |           |             |                  |      0.13s |       0.43s |                      | huggingface-hub |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |         |                   |                       |                |              |           |             |                  |      0.13s |       0.42s |                      | huggingface-hub |
| `mlx-community/GLM-4.6V-nvfp4`                          |         |                   |                       |                |              |           |             |                  |      0.13s |       0.42s |                      | huggingface-hub |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |         |                   |                       |                |              |           |             |                  |      0.22s |       0.52s |                      | huggingface-hub |
| `mlx-community/InternVL3-14B-8bit`                      |         |                   |                       |                |              |           |             |                  |      0.23s |       0.53s |                      | huggingface-hub |
| `mlx-community/InternVL3-8B-bf16`                       |         |                   |                       |                |              |           |             |                  |      0.13s |       0.43s |                      | huggingface-hub |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |         |                   |                       |                |              |           |             |                  |      0.13s |       0.43s |                      | huggingface-hub |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |         |                   |                       |                |              |           |             |                  |      0.15s |       0.43s |                      | huggingface-hub |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |         |                   |                       |                |              |           |             |                  |      0.17s |       0.45s |                      | huggingface-hub |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |         |                   |                       |                |              |           |             |                  |      0.16s |       0.44s |                      | huggingface-hub |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |         |                   |                       |                |              |           |             |                  |      0.17s |       0.45s |                      | huggingface-hub |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |         |                   |                       |                |              |           |             |                  |      0.12s |       0.42s |                      | huggingface-hub |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |         |                   |                       |                |              |           |             |                  |      0.22s |       0.52s |                      | huggingface-hub |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |         |                   |                       |                |              |           |             |                  |      0.20s |       0.51s |                      | huggingface-hub |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |         |                   |                       |                |              |           |             |                  |      0.22s |       0.52s |                      | huggingface-hub |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |         |                   |                       |                |              |           |             |                  |      0.13s |       0.43s |                      | huggingface-hub |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      0.15s |       0.50s |                      | huggingface-hub |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |         |                   |                       |                |              |           |             |                  |      0.13s |       0.57s |                      | huggingface-hub |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                   |                       |                |              |           |             |                  |      0.22s |       0.70s |                      | huggingface-hub |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |         |                   |                       |                |              |           |             |                  |      0.22s |       0.71s |                      | huggingface-hub |
| `mlx-community/Qwen3.5-27B-4bit`                        |         |                   |                       |                |              |           |             |                  |      0.15s |       0.69s |                      | huggingface-hub |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |         |                   |                       |                |              |           |             |                  |      0.20s |       0.71s |                      | huggingface-hub |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |         |                   |                       |                |              |           |             |                  |      0.19s |       0.68s |                      | huggingface-hub |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |         |                   |                       |                |              |           |             |                  |      0.19s |       0.70s |                      | huggingface-hub |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |         |                   |                       |                |              |           |             |                  |      0.11s |       0.59s |                      | huggingface-hub |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |         |                   |                       |                |              |           |             |                  |      0.11s |       0.59s |                      | huggingface-hub |
| `mlx-community/SmolVLM-Instruct-bf16`                   |         |                   |                       |                |              |           |             |                  |      0.13s |       0.60s |                      | huggingface-hub |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |         |                   |                       |                |              |           |             |                  |      0.16s |       0.68s |                      | huggingface-hub |
| `mlx-community/X-Reasoner-7B-8bit`                      |         |                   |                       |                |              |           |             |                  |      0.12s |       0.66s |                      | huggingface-hub |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |         |                   |                       |                |              |           |             |                  |      0.11s |       0.64s |                      | huggingface-hub |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |         |                   |                       |                |              |           |             |                  |      0.14s |       0.61s |                      | huggingface-hub |
| `mlx-community/gemma-3n-E2B-4bit`                       |         |                   |                       |                |              |           |             |                  |      0.22s |       0.60s |                      | huggingface-hub |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |         |                   |                       |                |              |           |             |                  |      0.20s |       0.51s |                      | huggingface-hub |
| `mlx-community/gemma-4-31b-bf16`                        |         |                   |                       |                |              |           |             |                  |      0.14s |       0.50s |                      | huggingface-hub |
| `mlx-community/gemma-4-31b-it-4bit`                     |         |                   |                       |                |              |           |             |                  |      0.23s |       0.54s |                      | huggingface-hub |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |         |                   |                       |                |              |           |             |                  |      0.17s |       0.45s |                      | huggingface-hub |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |         |                   |                       |                |              |           |             |                  |      0.17s |       0.45s |                      | huggingface-hub |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |         |                   |                       |                |              |           |             |                  |      0.17s |       0.45s |                      | huggingface-hub |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |         |                   |                       |                |              |           |             |                  |      0.16s |       0.45s |                      | huggingface-hub |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |         |                   |                       |                |              |           |             |                  |      0.16s |       0.44s |                      | huggingface-hub |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |         |                   |                       |                |              |           |             |                  |      0.10s |       0.38s |                      | huggingface-hub |
| `mlx-community/pixtral-12b-8bit`                        |         |                   |                       |                |              |           |             |                  |      0.13s |       0.41s |                      | huggingface-hub |
| `mlx-community/pixtral-12b-bf16`                        |         |                   |                       |                |              |           |             |                  |      0.17s |       0.45s |                      | huggingface-hub |
| `qnguyen3/nanoLLaVA`                                    |         |                   |                       |                |              |           |             |                  |      0.11s |       0.39s |                      | huggingface-hub |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,789 |                   500 |          2,289 |          370 |      49.1 |         5.5 |           15.88s |      0.69s |      17.05s | repetitive(unt), ... |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Automated review digest:_ [review.md](review.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.4.0
- _macOS Version:_ 26.4.1
- _SDK Version:_ 26.4
- _Xcode Version:_ 26.4.1
- _Xcode Build:_ 17E202
- _Metal SDK:_ MacOSX.sdk
- _Python Version:_ 3.13.12
- _Architecture:_ arm64
- _GPU/Chip:_ Apple M5 Max
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18

## Library Versions

- `numpy`: `2.4.4`
- `mlx`: `0.31.2.dev20260418+fa4320d5`
- `mlx-vlm`: `0.4.4`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.11.0`
- `transformers`: `5.5.4`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-19 00:43:13 BST_
