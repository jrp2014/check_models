# [mlx-vlm][encoding] Tokenizer space-marker artifacts (for example  ) appeared in output (about 139 occurrences) affecting 1 model(s)

## Summary

1 model(s) share a `encoding` signal that clusters under `mlx-vlm`.

- **Issue kind:** `harness`
- **Cluster ID:** `mlx-vlm_encoding_001`
- **Symptom family:** `encoding`
- **Acceptance signal:** Affected reruns contain no leaked BPE, byte-level, or tokenizer marker text.


## Affected Models

| Model                                                   | Representative Signal                                                                        | Token Context                                                                | Repro Bundle   |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|----------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 139 occurrences). | prompt=2,097 \| output/prompt=8.20% \| nontext burden=100% \| stop=completed |                |


## Evidence

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

Observed signals:

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 139 occurrences).

Sample output:

```text
TheĠimageĠdepictsĠaĠsereneĠoutdoorĠscene,ĠlikelyĠinĠaĠparkĠorĠgarden.ĠTheĠfocalĠpointĠisĠaĠpersonĠstandingĠonĠaĠsmall,ĠwoodenĠpierĠthatĠextendsĠintoĠaĠcalmĠbodyĠofĠwater.ĠTheĠindividualĠisĠdressedĠ...
```


## Likely Root Cause

- _Likely owner:_ `mlx-vlm`
- _Confidence:_ high
- _Issue kind:_ `harness`
- _Issue subtype:_ `encoding`
- _Why this classification is credible:_ Tokenizer space-marker artifacts (for
  example Ġ) appeared in output (about 139 occurrences). \| nontext prompt
  burden=100%
- _Suggested next action:_ Inspect decode cleanup; tokenizer markers are
  leaking into user-facing text.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit
```


## Fix Checklist

- [ ] Inspect tokenizer decode cleanup for byte-level/BPE marker leakage.
- [ ] Compare `decode` and `batch_decode` behavior with `skip_special_tokens=True`.
- [ ] Verify processor/tokenizer config does not require model-specific cleanup flags.


## Acceptance Criteria

- [ ] Affected reruns contain no leaked BPE, byte-level, or tokenizer marker text.
- [ ] The cluster rerun no longer produces this `mlx-vlm_encoding_001` maintainer-triage cluster.


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

