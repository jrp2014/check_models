# [mlx-vlm][stop-token] Special control token &lt;|end|&gt; appeared in generated text affecting 4 model(s)

## Summary

4 model(s) share a `stop_token` signal that clusters under `mlx-vlm`.

- **Issue kind:** `harness`
- **Cluster ID:** `mlx-vlm_stop-token_001`
- **Symptom family:** `stop_token`
- **Acceptance signal:** Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete.


## Affected Models

| Model                                | Representative Signal                                                                                                                        | Token Context                                                                                       | Repro Bundle   |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|----------------|
| `microsoft/Phi-3.5-vision-instruct`  | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. | prompt=768 \| output/prompt=65.10% \| nontext burden=99% \| stop=completed \| hit token cap (500)   |                |
| `mlx-community/GLM-4.6V-Flash-6bit`  | Special control token &lt;/think&gt; appeared in generated text.                                                                             | prompt=6,091 \| output/prompt=8.21% \| nontext burden=100% \| stop=completed \| hit token cap (500) |                |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | Special control token &lt;/think&gt; appeared in generated text.                                                                             | prompt=6,091 \| output/prompt=8.21% \| nontext burden=100% \| stop=completed \| hit token cap (500) |                |
| `mlx-community/GLM-4.6V-nvfp4`       | Special control token &lt;/think&gt; appeared in generated text.                                                                             | prompt=6,091 \| output/prompt=8.21% \| nontext burden=100% \| stop=completed \| hit token cap (500) |                |


## Evidence

### `microsoft/Phi-3.5-vision-instruct`

Observed signals:

- Special control token &lt;\|end\|&gt; appeared in generated text.
- Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

Sample output:

```text
The image shows a tranquil park scene with a person standing on a wooden dock, fishing by a pond. There are trees, a bench, and a small pine tree in the foreground. The weather appears to be overca...
```

### `mlx-community/GLM-4.6V-Flash-6bit`

Observed signals:

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;; Excessive markdown headers (6).
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

Sample output:

```text
<think>Got it, let's describe this picture in detail. First, the scene is a serene park or garden with a lake or pond. 

In the foreground, there's a small pine tree planted in a bed with dark soil...
```

### `mlx-community/GLM-4.6V-Flash-mxfp4`

Observed signals:

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;; Excessive markdown headers (6).
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

Sample output:

```text
<think>Got it, let's describe this picture in detail. First, the scene is a serene park or garden with a lake or pond. 

In the foreground, there's a large, moss-covered rock. Next to it, a small p...
```

_Additional affected models are listed in the Affected Models table above._


## Likely Root Cause

- _Likely owner:_ `mlx-vlm`
- _Confidence:_ high
- _Issue kind:_ `harness`
- _Issue subtype:_ `stop_token`
- _Why this classification is credible:_ Special control token &lt;\|end\|&gt;
  appeared in generated text. \| Special control token &lt;\|endoftext\|&gt;
  appeared in generated text. \| hit token cap (500) \| nontext prompt
  burden=99%
- _Suggested next action:_ Inspect EOS/stop-token stripping; control tokens
  are leaking into user-facing text.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Phi-3.5-vision-instruct mlx-community/GLM-4.6V-Flash-6bit mlx-community/GLM-4.6V-Flash-mxfp4 mlx-community/GLM-4.6V-nvfp4
```


## Fix Checklist

- [ ] Inspect model EOS token IDs and tokenizer special-token mappings.
- [ ] Verify mlx-vlm stop criteria receive all configured EOS/stop tokens.
- [ ] Check `skip_special_tokens` handling during decode.
- [ ] Strip generated control tokens such as `<|end|>` and `</think>` only after confirming generation stopped at the right boundary.


## Acceptance Criteria

- [ ] Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete.
- [ ] The cluster rerun no longer produces this `mlx-vlm_stop-token_001` maintainer-triage cluster.


## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260502+e8ebdebe |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.7.0                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.13.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.4.0               |
| macOS Version   | 26.4.1                      |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |

