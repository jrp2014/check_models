<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 1 failure(s), 2 harness issue(s), 0 text-sanity issue(s) (mlx-vlm 0.6.5)

**Run summary:** 61 locally-cached VLM model(s) checked; 1 hard failure(s), 2 harness/integration issue(s), 0 text-sanity/semantic issue(s), 0 preflight warning(s), 60 successful run(s).

Test image: `cats.jpg` (0.2 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

<!-- markdownlint-disable MD060 -->

| Target                                         | Problem                                                | Evidence Snapshot                                                                                                                                                                  | Affected Models                              | Issue Draft                                                                                                                      | Evidence Bundle                                                                                                                                                                         | Fixed When                                                |
|------------------------------------------------|--------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx-vlm`                                      | mlx-vlm: Decode / model error: list index out of range | Model Error \| phase decode \| IndexError                                                                                                                                          | 1: `mlx-community/gemma-4-31b-bf16`          | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx-vlm_mlx-vlm-decode-model_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T203149Z_002_mlx-community_gemma-4-31b-bf16_MLX_VLM_DECODE_MODEL_0375357f22c1.json)     | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | Stop/control tokens leaked into generated text         | decoded text contains control token &lt;/think&gt; \| prompt=317 \| output/prompt=59.31% \| nontext burden=98% \| stop=completed                                                   | 1: `mlx-community/Qwen3-VL-2B-Thinking-bf16` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_stop-token_001.md)           | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T203149Z_001_mlx-community_Qwen3-VL-2B-Thinking-bf16_mlx_vlm_stop_token_001.json)       | No leaked stop/control tokens.                            |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short  | generated_tokens~3 \| prompt_tokens=4103, output_tokens=3, output/prompt=0.1%, weak text=truncated \| prompt=4,103 \| output/prompt=0.07% \| nontext burden=100% \| stop=completed | 1: `mlx-community/paligemma2-3b-pt-896-4bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm-mlx_long-context_001.md)     | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T203149Z_003_mlx-community_paligemma2-3b-pt-896-4bit_mlx_vlm_mlx_long_context_001.json) | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

---

## 1. Failure affecting 1 model

**Affected models:** `mlx-community/gemma-4-31b-bf16`

**Observed error:**

```text
Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
```

**Relevant upstream traceback:**

```text
    detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 171, in add_token
    if self.is_byte_token[token]:
       ~~~~~~~~~~~~~~~~~~^^^^^^^
IndexError: list index out of range
```

<!-- markdownlint-disable MD060 -->

| Model                            | Observed Behavior                                                                                                                                                                              | First Seen Failing      | Recent Repro           |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/gemma-4-31b-bf16` | Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory). | 2026-04-10 14:13:07 BST | 2/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/gemma-4-31b-bf16`

---

## Harness/Integration Issues (2 model(s))

2 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.

**Sample output:**

```text
...lements are the two cats, the remotes, and the pink fabric. I need to describe this briefly.
</think>

Two tabby cats are lying on a bright pink couch, relaxed and asleep. A white remote control...
```

### `mlx-community/paligemma2-3b-pt-896-4bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.
- At long prompt length (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%; weak text signal truncated).

**Sample output:**

```text
Cat.
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** `mlx-community/gemma-4-31b-it-4bit`
**Generation regressions in history window:** `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`
**Quality regressions in history window:** `mlx-community/gemma-4-31b-bf16`
**Core library changes in window:** `mlx=0.32.0.dev20260620+c9e8ee6b->0.32.1.dev20260710+4367c73b`, `mlx-vlm=0.6.3->0.6.5`, `transformers=5.12.1->5.13.0`

<!-- markdownlint-disable MD060 -->

| Model                            | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|----------------------------------|--------------------------|-------------------------|------------------------|
| `mlx-community/gemma-4-31b-bf16` | still failing            | 2026-04-10 14:13:07 BST | 2/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 3
- **Summary diagnostics models:** 58
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 535.21s (535.21s)
- **Average runtime per model:** 8.77s (8.77s)
- **Dominant runtime phase:** post-prefill decode dominated 19/61 measured model runs (38% of tracked runtime).
- **Phase totals:** model load=132.35s, local prompt prep=0.21s, upstream prefill / first-token=47.82s, post-prefill decode=206.16s, generation total (unsplit)=148.26s, cleanup=7.69s
- **Generation total:** 402.25s across 61 model(s); upstream prefill / first-token split available for 60/61 model(s).
- **Observed stop reasons:** completed=52, exception=1, max_tokens=8
- **Validation overhead:** 0.11s total (avg 0.00s across 61 model(s)).
- **Upstream prefill / first-token latency:** Avg 0.80s | Min 0.02s | Max 5.02s across 60 model(s).
- **What this likely means:** Most measured runtime is spent generating after the first token is available.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (58 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 48 model(s).
- **Ran, but with quality warnings:** 10 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg --trust-remote-code --max-tokens 200 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --presence-context-size 20 --frequency-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `mlx-community/gemma-4-31b-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg`
- Generation settings: max_tokens=200, temperature=0.0, top_p=1.0

Report generated on: 2026-07-10 21:31:49 BST by [check_models](https://github.com/jrp2014/check_models).

<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.5                                                                                                                                                    |
| mlx                        | 0.32.1.dev20260710+4367c73b                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.5                                                                                                                                                    |
| transformers               | 5.13.0                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.23.0                                                                                                                                                   |
| Python Version             | 3.13.13                                                                                                                                                  |
| OS                         | Darwin 25.5.0                                                                                                                                            |
| macOS Version              | 26.5.2                                                                                                                                                   |
| SDK Version                | 26.5                                                                                                                                                     |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                                       |
| Xcode Version              | 26.6                                                                                                                                                     |
| Xcode Build                | 17F113                                                                                                                                                   |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                               |
| Metal SDK                  | MacOSX26.5.sdk                                                                                                                                           |
| Metal Compiler Version     | Apple metal version 32023.883 (metalfe-32023.883)                                                                                                        |
| Metallib Linker Version    | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                           |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                          |
| GPU/Chip                   | Apple M5 Max                                                                                                                                             |
| GPU Cores                  | 40                                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                                  |
| MLX Install Type           | editable local source                                                                                                                                    |
| MLX Distribution Root      | /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                                       |
| MLX Core Extension         | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,449,848 bytes, sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=e61c827cd79f978aa5eacc136f65d6dea065005787f3a1457dc9d4512d6ee9cf)  |
| RAM                        | 128.0 GB                                                                                                                                                 |
<!-- markdownlint-enable MD060 -->
