# \[mlx-vlm\]\[mlx-vlm: Decode / runtime error\] mlx-vlm: Decode / runtime error: property 'text' of 'NaiveStreamingDetokenizer' object has no affecting 3 model(s)

## Summary

3 model(s) show **mlx-vlm: Decode / runtime error** that should be filed against mlx-vlm.

- **Observed problem:** mlx-vlm: Decode / runtime error: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
- **Target:** mlx-vlm
- **Raw owner hint:** `mlx-vlm`
- **Affected models:** 3
- **Fixed when:** Load/generation completes or fails with a narrower owner.
- **Issue kind:** `runtime_failure`
- **Raw cluster:** `mlx-vlm_mlx-vlm-decode-error_001` (`MLX_VLM_DECODE_ERROR`)


## Affected Models

| Model                                              | Representative Signal                                                | Token Context   | Repro JSON                                                                                                                                  |
|----------------------------------------------------|----------------------------------------------------------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` | instruction echo \| metadata borrowing \| hallucination              | stop=exception  | [repro JSON](../repro_bundles/20260508T130439Z_004_mlx-community_ERNIE-4.5-VL-28B-A3B-Thinking-bf16_MLX_VLM_DECODE_ERROR_c6f291b6246e.json) |
| `mlx-community/LFM2-VL-1.6B-8bit`                  | Special control token &lt;\|im_end\|&gt; appeared in generated text. | stop=exception  | [repro JSON](../repro_bundles/20260508T130439Z_006_mlx-community_LFM2-VL-1.6B-8bit_MLX_VLM_DECODE_ERROR_94132d05282d.json)                  |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                | Special control token &lt;\|im_end\|&gt; appeared in generated text. | stop=exception  | [repro JSON](../repro_bundles/20260508T130439Z_007_mlx-community_LFM2.5-VL-1.6B-bf16_MLX_VLM_DECODE_ERROR_0b3df5c094f3.json)                |


## Minimal Evidence

- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` fails with: Model runtime error during generation for mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
- Root exception: `builtins.AttributeError`: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
- `mlx-community/LFM2-VL-1.6B-8bit` fails with: Model runtime error during generation for mlx-community/LFM2-VL-1.6B-8bit: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
- Root exception: `builtins.AttributeError`: property 'text' of 'NaiveStreamingDetokenizer' object has no setter


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 mlx-community/LFM2-VL-1.6B-8bit mlx-community/LFM2.5-VL-1.6B-bf16
```

Repro bundles:

- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: [repro JSON](../repro_bundles/20260508T130439Z_004_mlx-community_ERNIE-4.5-VL-28B-A3B-Thinking-bf16_MLX_VLM_DECODE_ERROR_c6f291b6246e.json)
- `mlx-community/LFM2-VL-1.6B-8bit`: [repro JSON](../repro_bundles/20260508T130439Z_006_mlx-community_LFM2-VL-1.6B-8bit_MLX_VLM_DECODE_ERROR_94132d05282d.json)
- `mlx-community/LFM2.5-VL-1.6B-bf16`: [repro JSON](../repro_bundles/20260508T130439Z_007_mlx-community_LFM2.5-VL-1.6B-bf16_MLX_VLM_DECODE_ERROR_0b3df5c094f3.json)
- Note: these are local artifact links; attach or publish the JSON when filing upstream.

Per-model failure reruns:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/LFM2-VL-1.6B-8bit
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/LFM2.5-VL-1.6B-bf16
```


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The cluster rerun no longer produces this `mlx-vlm_mlx-vlm-decode-error_001` maintainer-triage cluster.


## Fix Checklist

- [ ] Inspect the exported error package, load phase, and traceback owner.
- [ ] Check model config, tokenizer files, and weight shape compatibility.
- [ ] Compare against installed mlx, mlx-vlm, mlx-lm, transformers, and tokenizers versions.
- [ ] Reproduce with the single affected model before judging output quality.


## Likely Root Cause

- _Filing target:_ mlx-vlm
- _Likely owner:_ `mlx-vlm`
- _Confidence:_ high
- _Issue kind:_ `runtime_failure`
- _Issue subtype:_ `MLX_VLM_DECODE_ERROR`
- _Why this classification is credible:_ keywords=37 \| context echo=100% \|
  nonvisual metadata reused \| reasoning leak
- _Suggested next action:_ Treat as a model-quality limitation for this prompt
  and image.


## Appendix: Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260508+a1c0b6f9 |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.8.0.dev0                  |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.14.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.4.0               |
| macOS Version   | 26.4.1                      |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |


## Appendix: Detailed Evidence

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

Observed error:

```text
Model runtime error during generation for mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
```

Root exception:

```text
builtins.AttributeError: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
```

Traceback tail:

```text
    setattr(y, key, value)
    ~~~~~~~^^^^^^^^^^^^^^^
AttributeError: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model runtime error during generation for mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
```

### `mlx-community/LFM2-VL-1.6B-8bit`

Observed error:

```text
Model runtime error during generation for mlx-community/LFM2-VL-1.6B-8bit: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
```

Root exception:

```text
builtins.AttributeError: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
```

Traceback tail:

```text
    setattr(y, key, value)
    ~~~~~~~^^^^^^^^^^^^^^^
AttributeError: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model runtime error during generation for mlx-community/LFM2-VL-1.6B-8bit: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
```

### `mlx-community/LFM2.5-VL-1.6B-bf16`

Observed error:

```text
Model runtime error during generation for mlx-community/LFM2.5-VL-1.6B-bf16: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
```

Root exception:

```text
builtins.AttributeError: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
```

Traceback tail:

```text
    setattr(y, key, value)
    ~~~~~~~^^^^^^^^^^^^^^^
AttributeError: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model runtime error during generation for mlx-community/LFM2.5-VL-1.6B-bf16: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
```

