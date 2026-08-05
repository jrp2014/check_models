# Self-contained Reproduction Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pasted crash reports disclose exact reproduction inputs without referring to private or nonexistent files.

**Architecture:** Add an optional metadata-only public image URL to the CLI and publication-safe run manifest. Normalize retained image facts into one typed record, then use shared report helpers to render either a complete public-image native reproduction or an explicit local-only input description in both aggregate and per-crash issue reports.

**Tech Stack:** Python 3.13, argparse, urllib.parse, Pillow image facts, typed report blocks, pytest, Ruff, mypy/ty/Pyrefly.

## Global Constraints

- `--image-source-url` records metadata only; it never downloads or replaces the inference input.
- Only absolute HTTP(S) source URLs are accepted.
- Local filesystem paths remain excluded from publication-safe artifacts.
- Local-only reports describe the image but do not claim to provide a runnable reproduction.
- Paste-ready report paths must not refer to synthetic `reproduce.py` or `prompt.txt` files.
- Tests write only under pytest temporary directories; tracked `src/output/` artifacts are not regenerated.
- Keep `src/check_models.py` as one file.

---

### Task 1: Public source URL contract

**Files:**

- Modify: `src/check_models.py:1014-1049, 14586-14665, 16266-16280, 17127-17152`
- Test: `src/tests/test_parameter_validation.py`
- Test: `src/tests/test_jsonl_output.py`

**Interfaces:**

- Consumes: the existing `_build_cli_parser()`, `validate_cli_arguments()`, `_run_image_record()`, and `save_run_json_report()` paths.
- Produces: `_parse_public_image_source_url(value: str) -> str`, CLI attribute `args.image_source_url: str | None`, optional `RunImageRecord["source_url"]`, and `save_run_json_report(..., image_source_url: str | None = None)`.

- [ ] **Step 1: Write failing parser and run-manifest tests**

```python
def test_image_source_url_accepts_absolute_https_metadata():
    parser = check_models._build_cli_parser()
    args = parser.parse_args(["--image", "local.jpg", "--image-source-url", "https://example.test/cats.jpg"])
    assert args.image_source_url == "https://example.test/cats.jpg"


@pytest.mark.parametrize("value", ["cats.jpg", "file:///tmp/cats.jpg", "ftp://example.test/cats.jpg"])
def test_image_source_url_rejects_non_public_sources(value):
    parser = check_models._build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--image-source-url", value])
```

Extend `test_save_run_json_report_captures_public_snapshot_contract` to pass
`image_source_url="https://example.test/cats.jpg"` and assert that exact value
is stored under `payload["image"]["source_url"]` while the local parent path is
absent from the serialized JSON.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
conda activate mlx-vlm
pytest src/tests/test_parameter_validation.py src/tests/test_jsonl_output.py -q
```

Expected: FAIL because the flag and keyword parameter do not exist.

- [ ] **Step 3: Implement the minimal CLI and manifest plumbing**

Implement `_parse_public_image_source_url` with `urlparse`, accepting only
`http`/`https` plus a non-empty network location and raising
`argparse.ArgumentTypeError` otherwise. Register `--image-source-url` in the
Input argument group outside the folder/image mutex. Add
`source_url: NotRequired[str]` to `RunImageRecord`; have `_run_image_record`
include it only when non-null; thread `getattr(inputs.run_args,
"image_source_url", None)` through `save_run_json_report`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit the input contract**

```bash
git add src/check_models.py src/tests/test_parameter_validation.py src/tests/test_jsonl_output.py
git commit -m "feat: record public reproduction image sources"
```

### Task 2: Honest paste-ready reproduction blocks

**Files:**

- Modify: `src/check_models.py:1114-1120, 14851-15165, 15404-16055`
- Test: `src/tests/test_report_generation.py`

**Interfaces:**

- Consumes: `RunImageRecord`, `JsonlMetadataRecord["prompt"]`, existing native mlx-vlm CLI builders, resolved model provenance, and retained generation settings.
- Produces: `RunIssueSummarySource.image: RunImageRecord | None`; validation/narrowing for retained image facts; shared report blocks describing image facts and exact prompt; a runnable public-source reproduction block; and a non-runnable local-only explanation.

- [ ] **Step 1: Write failing aggregate-report tests**

Extend `_write_issue_summary_fixture` with an optional complete image record.
For a public source, assert the crash section includes the HTTP URL, dimensions,
byte size, SHA-256, exact prompt, `curl --fail --location`, integrity checking,
`python -m mlx_vlm.generate`, and the resolved revision. Assert it excludes
`reproduce.py` and `prompt.txt`. For a local-only source, assert it says the
original local input is not published, includes the recorded characteristics,
and excludes the local basename/path from command blocks.

- [ ] **Step 2: Write failing direct-issue-draft tests**

Generate a crash draft once with `Namespace(image_source_url="https://example.test/cats.jpg", ...)`
and once with `image_source_url=None`. Assert the public draft contains the
download/integrity/native command sequence and exact prompt. Assert the local
draft contains only the unpublished-input explanation and image facts, with no
plausible runnable reference to the private fixture.

- [ ] **Step 3: Run focused report tests and verify RED**

Run:

```bash
conda activate mlx-vlm
pytest src/tests/test_report_generation.py -q
```

Expected: FAIL on the new public/local reproduction assertions.

- [ ] **Step 4: Implement retained image validation and shared rendering**

Change `_load_run_issue_enrichment` to narrow the optional image mapping into a
publication-safe `RunImageRecord`, preserving valid characteristics while
discarding an invalid optional URL. Add helpers that format the suffix as an
image format, dimensions, bytes, and digest; render the exact prompt in a
collapsible `ReportDetails`; and return one of:

```text
Public source URL + curl download + SHA-256 check + native mlx-vlm command
```

or:

```text
The original local input is not published, so this report does not claim a complete reproduction command.
```

Use a downloaded basename only in the public branch. Build the native command
with the retained prompt, resolved revision, and parsed retained generation
settings. Replace `_run_issue_summary_repro_command` and the direct issue
draft's unconditional CLI/Python blocks with the shared distinction.

- [ ] **Step 5: Run focused report tests and verify GREEN**

Run the command from Step 3. Expected: PASS.

- [ ] **Step 6: Commit report rendering**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "fix: make crash reproduction inputs self-contained"
```

### Task 3: User documentation and full verification

**Files:**

- Modify: `src/README.md:1050-1060`
- Modify: `CHANGELOG.md:6-35`

**Interfaces:**

- Consumes: the completed CLI/report contract from Tasks 1 and 2.
- Produces: user-facing guidance for `--image-source-url` and an `[Unreleased]` changelog entry.

- [ ] **Step 1: Document the metadata-only flag and report fallback**

Add the CLI reference row:

```markdown
| `--image-source-url` | URL | omitted | Public HTTP(S) location of the exact local image used by the run; recorded for issue reproduction only and never downloaded as the inference input. |
```

Explain that omitted URLs cause issue drafts to publish characteristics and an
unavailable-original notice instead of a misleading command.

- [ ] **Step 2: Update the changelog**

Under `[Unreleased] / Changed`, record that crash reports now inline exact
prompts, use public image download/integrity instructions when supplied, and
describe unpublished local images without synthetic reproduction files.

- [ ] **Step 3: Format and clear safe lint findings**

Run:

```bash
conda activate mlx-vlm
make format
make -C src lint-fix
make lint
```

Expected: all commands exit 0. Do not retain unsafe Ruff fixes unless they are
faster than a manual correction and each semantic diff is critically reviewed.

- [ ] **Step 4: Run the full quality gate**

Run:

```bash
conda activate mlx-vlm
make quality
```

Expected: exit 0, with no tracked `src/output/` modifications.

- [ ] **Step 5: Review the final diff and commit documentation**

Run `git diff --check`, inspect `git diff --stat` and `git status --short`, then:

```bash
git add src/README.md CHANGELOG.md
git commit -m "docs: explain public reproduction image metadata"
```
