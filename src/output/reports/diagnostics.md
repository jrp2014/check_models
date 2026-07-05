<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 1 failure(s), 8 harness issue(s) (mlx-vlm 0.6.4)

**Run summary:** 61 locally-cached VLM model(s) checked; 1 hard failure(s), 8 harness/integration issue(s), 0 preflight warning(s), 60 successful run(s).

Test image: `cats.jpg` (0.2 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                                                             | Evidence Snapshot                                                                                                                                                                  | Affected Models                                            | Issue Draft                                                                                                                                          | Evidence Bundle                                                                                                                                                                                                   | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `huggingface_hub`                                        | Hugging Face Hub: Model load / model error: Operation timed out after 300.0 seconds | Model Error \| phase model_load \| TimeoutError                                                                                                                                    | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_huggingface-hub_huggingface-hub-model-load-model_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_HUGGINGFACE_HUB_MODEL_LOAD_MODEL_6c3197f.json) | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text                                      | decoded text contains control token &lt;\|end\|&gt; \| prompt=1,330 \| output/prompt=13.08% \| nontext burden=100% \| stop=completed \| 2 model cluster                            | 2: `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (+1)    | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_stop-token_001.md)                               | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_002_mlx-community_Apriel-1.5-15b-Thinker-6bit-MLX_mlx_vlm_stop_token_001.json)                        | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                                               | generated_tokens~3 \| prompt=315 \| output/prompt=0.95% \| nontext burden=98% \| stop=completed \| 5 model cluster                                                                 | 5: `Qwen/Qwen3-VL-2B-Instruct` (+4)                        | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_model-config-mlx-vlm_prompt-template_001.md)             | [5 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_001_Qwen_Qwen3-VL-2B-Instruct_model_config_mlx_vlm_prompt_template_001.json)                          | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short                               | generated_tokens~3 \| prompt_tokens=4103, output_tokens=3, output/prompt=0.1%, weak text=truncated \| prompt=4,103 \| output/prompt=0.07% \| nontext burden=100% \| stop=completed | 1: `mlx-community/paligemma2-3b-pt-896-4bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm-mlx_long-context_001.md)                         | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260705T003436Z_009_mlx-community_paligemma2-3b-pt-896-4bit_mlx_vlm_mlx_long_context_001.json)                           | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

---

## 1. Failure affecting 1 model

**Affected models:** `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**Observed error:**

```text
Model loading failed: Operation timed out after 300.0 seconds
```

**Relevant upstream traceback:**

```text
    self._condition.wait(timeout)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/threading.py", line 359, in wait
    waiter.acquire()
    ~~~~~~~~~~~~~~^^
TimeoutError: Operation timed out after 300.0 seconds
```

<!-- markdownlint-disable MD060 -->

| Model                                                   | Observed Behavior                                             | First Seen Failing      | Recent Repro           |
|---------------------------------------------------------|---------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | Model loading failed: Operation timed out after 300.0 seconds | 2026-04-19 00:39:44 BST | 1/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

---

## Harness/Integration Issues (8 model(s))

8 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `Qwen/Qwen3-VL-2B-Instruct`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.

**Sample output:**

```text
This image
```

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).
- Output leaked reasoning or prompt-template text (here are my reasoning steps, the user asks:).

**Sample output:**

```text
...at's it. No need for extra. We'll comply.
[BEGIN FINAL RESPONSE]
Two tabby cats are curled up sleeping side‑by‑side on a pink couch, with a TV remote resting nearby.
[END FINAL RESPONSE]
<|end|>
```

### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.

**Sample output:**

```text
This image
```

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.

**Sample output:**

```text
So,,
</think>

Two cats are lying on a bright pink blanket. One cat is a tabby with darker stripes, and the other is a calico with a mix of orange, black, and white fur. Both cats are relaxed, w...
```

### `mlx-community/Qwen3.5-35B-A3B-bf16`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.

**Sample output:**

```text
os,
```

### `mlx-community/Qwen3.6-27B-mxfp8`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 2 tokens.

**Sample output:**

```text
.
```

### `mlx-community/X-Reasoner-7B-8bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.

**Sample output:**

```text
The image
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

**Regressions since previous run:** `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`
**Recoveries since previous run:** `mlx-community/MiniCPM-V-4.6-8bit`, `mlx-community/diffusiongemma-26B-A4B-it-8bit`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`
**Quality regressions in history window:** `Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/Qwen3-VL-2B-Instruct-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3.5-35B-A3B-bf16`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/X-Reasoner-7B-8bit`
**Core library changes in window:** `mlx=0.32.0.dev20260614+89064477->0.32.0.dev20260704+de7b4ed9`, `mlx-vlm=0.6.3->0.6.4`, `transformers=5.12.0->5.12.1`

<!-- markdownlint-disable MD060 -->

| Model                                                   | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|---------------------------------------------------------|--------------------------|-------------------------|------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | new regression           | 2026-04-19 00:39:44 BST | 1/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 9
- **Summary diagnostics models:** 52
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 886.07s (886.07s)
- **Average runtime per model:** 14.53s (14.53s)
- **Dominant runtime phase:** model load dominated 38/61 measured model runs (67% of tracked runtime).
- **Phase totals:** model load=597.46s, local prompt prep=0.23s, upstream prefill / first-token=63.86s, post-prefill decode=224.04s, cleanup=7.31s
- **Generation total:** 287.90s across 60 model(s); upstream prefill / first-token split available for 60/60 model(s).
- **Observed stop reasons:** completed=53, exception=1, max_tokens=7
- **Validation overhead:** 0.11s total (avg 0.00s across 61 model(s)).
- **Upstream prefill / first-token latency:** Avg 1.06s | Min 0.02s | Max 6.11s across 60 model(s).
- **What this likely means:** Cold model load time is a major share of runtime for this cohort.
- **Suggested next action:** Consider staged runs, model reuse, or narrowing the model set before reruns.

---

## Models Not Flagged (52 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 43 model(s).
- **Ran, but with quality warnings:** 9 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg --trust-remote-code --max-tokens 200 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --presence-context-size 20 --frequency-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`, `mlx-community/Qwen3-VL-2B-Instruct-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3.5-35B-A3B-bf16`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/paligemma2-3b-pt-896-4bit`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg`
- Generation settings: max_tokens=200, temperature=0.0, top_p=1.0

Report generated on: 2026-07-05 01:34:36 BST by [check_models](https://github.com/jrp2014/check_models).

<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.4                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260704+de7b4ed9                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
| transformers               | 5.12.1                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.22.0                                                                                                                                                   |
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
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,451,352 bytes, sha256=7e5c9a3a3225bf3b04a5fe67c50602975d3698a45e2113433465848af47fd70c) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,747,136 bytes, sha256=9d942d98a9a9f3e42b3f22c6606bc1ee621d28a9fb512d0cdba6edbb9ef79df8)  |
| RAM                        | 128.0 GB                                                                                                                                                 |
<!-- markdownlint-enable MD060 -->
