# Self-contained Reproduction Inputs Design

## Goal

Make crash reports intelligible and honest when pasted into a GitHub issue. A
reader must be able to tell whether the exact source image is publicly
obtainable, see the exact prompt, and avoid commands that depend on files the
report does not provide.

## Input contract

Add an optional `--image-source-url URL` argument. It records where another
person can download the exact image used by the run; it does not download the
image or change the inference input selected by `--image` or `--folder`.

Only absolute HTTP(S) URLs are accepted. The value is stored in the
publication-safe `run.json` image record alongside the existing basename,
dimensions, byte size, and SHA-256 digest. Local filesystem paths remain
excluded.

## Report behaviour

Every expanded crash entry exposes:

- the exact prompt;
- image format, dimensions, byte size, and SHA-256 when recorded;
- the resolved model revision and generation settings already retained; and
- links to the detailed evidence artifacts.

When a public source URL is recorded, the report supplies a download command,
an integrity check, and a native `python -m mlx_vlm.generate` command using the
downloaded file and inline prompt.

When no public source URL is recorded, the report states that the original
local input is not published. It describes the image characteristics but does
not render its basename or path as a runnable input and does not claim to offer
a complete reproduction command.

The same distinction applies to direct per-crash issue drafts. Existing
synthetic references to `reproduce.py` and `prompt.txt` are removed from the
paste-ready paths.

## Compatibility and validation

The new `run.json` field is optional, so older retained runs still regenerate.
Malformed optional image enrichment is ignored rather than blocking summary
generation, matching the current retained-report policy.

Tests use temporary output directories and cover public and local-only inputs,
URL validation, JSON serialization, prompt inclusion, native command output,
and absence of private/synthetic file references. Tracked `src/output/`
artifacts are not regenerated.
