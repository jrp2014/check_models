# Model Output Gallery

Generated on: 2026-07-25 23:08:08 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Current-run Chooser

Current-run usability and captured resource facts only. Throughput requires at least 16 generated tokens.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                             | Usability   | Gen TPS    |   Peak GB |   Gen tok | Observations                          | Output preview                                                                                                                                                                                                                                                                           |
|---------------------------------------------------|-------------|------------|-----------|-----------|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava) | `unusable`  | 95.6 tok/s |       3.9 |       100 | repeated output, token cap truncation | The image primarily features a house with a red roof and a red roof. The house is surrounded by a green grassy field. The house has a red roof and a red roof. The house also has a red roof. The house has a red roof. The house has a red roof. The house has a red roof. The house... |
<!-- markdownlint-enable MD034 MD049 -->

## Avoid for This Run

<!-- markdownlint-disable MD034 MD049 -->

| Model                                             | Usability   | Observations                          | Output preview                                                                                                                                                                                                                                                                           |
|---------------------------------------------------|-------------|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava) | `unusable`  | repeated output, token cap truncation | The image primarily features a house with a red roof and a red roof. The house is surrounded by a green grassy field. The house has a red roof and a red roof. The house also has a red roof. The house has a red roof. The house has a red roof. The house has a red roof. The house... |
<!-- markdownlint-enable MD034 MD049 -->

## Lowest-memory Usable Models

No usable models in this run.

## Fastest Valid Generation

No valid throughput samples in this run.

No usable models in this run.

## Run Stamps

- `mlx-vlm`: `0.6.8`
- `mlx`: `0.32.1.dev20260725+973e27f82`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.24.0`
- _Python Version:_ 3.13.13
- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.2
- _GPU/Chip:_ Apple M5 Max
- _MLX Device:_ Apple M5 Max
- _GPU Architecture:_ applegpu_g17s
- _RAM:_ 128.0 GB
- _Recommended Working Set:_ 108 GB
- _Fused Attention:_ Available

## Image Metadata

- _Date:_ 2026-07-25 23:04:54 BST
- _Time:_ 23:04:54

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe the main elements in this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Complete Per-model Evidence

Complete generated or crash evidence for every attempted model.

<a id="model-qnguyen3-nanollava"></a>

### qnguyen3/nanoLLaVA

<details>
<summary>Complete evidence: qnguyen3/nanoLLaVA</summary>

- _Execution:_ completed
- _Usability:_ unusable
- _Maintainer status:_ observation_needs_reproduction
- _Observations:_ repeated output, token cap truncation
- _Model load time:_ 0.88s
- _Generation time:_ 1.33s
- _Total time:_ 2.27s
- _Input validation time:_ 0.0038
- _Prompt preparation time:_ 0.0565
- _First-token latency:_ 0.203
- _Cleanup time:_ 0.0699
- _Prompt tokens:_ 26
- _Generation tokens:_ 100
- _Total tokens:_ 126
- _Prompt throughput (raw):_ 128 tok/s
- _Generation throughput (raw):_ 95.6 tok/s
- _Peak memory:_ 3.9
- _Active memory:_ 2.1
- _Cache memory:_ 1.6
- _Model-load active memory:_ 2.1
- _Stop reason:_ max_tokens
- _Requested maximum tokens:_ 100
- _Rendered prompt characters:_ 156
- _Image placeholders:_ 1
- _Processed image:_ not captured
- _Image patch count:_ not captured
- _Processor:_ transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- _Tokenizer:_ transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- _Model revision:_ not captured by this result
- _Generation settings:_ {"max_tokens": 100, "prefill_step_size": 4096,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- _EOS token:_ <\|im_end\|>

_Complete generated output:_

```text
The image primarily features a house with a red roof and a red roof. The house is surrounded by a green grassy field. The house has a red roof and a red roof. The house also has a red roof. The house has a red roof. The house has a red roof. The house has a red roof. The house has a red roof. The house has a red roof. The house has a red roof. The house has a red roof. The house has a red roof
```

</details>

---
