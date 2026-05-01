# [Bug] HUGGINGFACE_HUB_MODEL_LOAD_MODEL:f6c345b68e53

## Description

A runtime failure occurred affecting **1 model(s)**.

### Affected Models

- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## At a Glance

- _Observed:_ Model loading failed: Server disconnected without sending a
  response.
- _Likely owner:_ `huggingface_hub`
- _Why it matters:_ This prevented a complete model response; stage `Model
  Error`; phase `model_load`; code `HUGGINGFACE_HUB_MODEL_LOAD_MODEL`; type
  `ValueError`.
- _Suggested next step:_ check cache/revision availability and network/auth
  state; Hub disconnects may be transient outages rather than model defects.
- _Affected models:_ `mlx-community/SmolVLM2-2.2B-Instruct-mlx`


## Maintainer Triage

- _Likely owner:_ huggingface-hub \| confidence=high
- _Classification:_ runtime_failure \| HUGGINGFACE_HUB_MODEL_LOAD_MODEL
- _Summary:_ model error \| huggingface hub model load model \| hub
  connectivity
- _Evidence:_ model error \| huggingface hub model load model \| hub
  connectivity
- _Token context:_ stop=exception
- _Next action:_ Check whether Hugging Face was reachable; this may be a
  transient Hub/network outage or disconnect rather than a model defect.


## Traceback / Error Message

```text
Model loading failed: Server disconnected without sending a response.
```

## Reproducibility

A reproduction bundle is available at: `20260501T231046Z_003_mlx-community_SmolVLM2-2.2B-Instruct-mlx_HUGGINGFACE_HUB_MODEL_LOAD_MODEL_f6c345b.json`

### Repro Command

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt Describe this picture --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/SmolVLM2-2.2B-Instruct-mlx
```

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260501+e8ebdebe |
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

