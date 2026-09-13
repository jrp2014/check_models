"""Pre-download check for Hugging Face hub candidates.

Before spending tens of gigabytes on a checkpoint, judge it from a few small
hub reads the way ``check_models`` will judge it once cached:

1. **Layout**: the server-style file rule (``config.json``,
   ``tokenizer_config.json``, safetensors weights).
2. **Architecture**: the ``model_type`` resolves (via ``MODEL_REMAPPING``) to
   an installed ``mlx_vlm/models/`` package. Folder-name check only.
3. **Chat template shape**: whether the template iterates message content
   parts (multimodal) or concatenates content as a string (text-only), which
   fails at prefill for vision families that send list-content messages.

Only ``config.json`` and the chat template are fetched, over HTTPS, without
touching the local Hugging Face cache. Verdicts are hints, never proof.

Usage::

    python -m tools.hub_precheck mlx-community/Some-VLM-4bit [more repos...]

Exit status 1 when any candidate is blocked.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal
from urllib.request import Request, urlopen

_LAYOUT_REQUIRED: Final[tuple[str, ...]] = ("config.json", "tokenizer_config.json")
_FETCH_TIMEOUT_S: Final[int] = 30

type TemplateShape = Literal["iterates-content", "string-only", "absent", "unknown"]

# A template that walks content parts, or branches on whether content is a
# string, can render list-content messages.
_CONTENT_ITERATION_RE: Final[re.Pattern[str]] = re.compile(
    r"for\s+\w+\s+in\s+message(?:\[['\"]content['\"]\]|\.content)"
    r"|content\s+is\s+(?:not\s+)?(?:string|iterable|sequence|mapping)"
    r"|\[['\"]type['\"]\]\s*(?:==|in\b)"
    r"|\.type\s*(?:==|in\b)"
)
# A template that only ever concatenates the content as a string cannot.
_CONTENT_CONCAT_RE: Final[re.Pattern[str]] = re.compile(
    r"\+\s*message(?:\[['\"]content['\"]\]|\.content)"
    r"|message(?:\[['\"]content['\"]\]|\.content)\s*\+"
)


def template_shape(template: str | None) -> TemplateShape:
    """Classify a chat template by how it handles message content."""
    if not template or not template.strip():
        return "absent"
    if _CONTENT_ITERATION_RE.search(template):
        return "iterates-content"
    if _CONTENT_CONCAT_RE.search(template):
        return "string-only"
    return "unknown"


def layout_missing(files: Iterable[str]) -> list[str]:
    """Return what the server-style cache-layout rule would find missing."""
    names = set(files)
    missing = [name for name in _LAYOUT_REQUIRED if name not in names]
    if "model.safetensors.index.json" not in names and not any(
        name.endswith(".safetensors") for name in names
    ):
        missing.append("*.safetensors")
    return missing


@dataclass(frozen=True)
class HubCandidate:
    """What the hub reads established about one candidate repo."""

    repo_id: str
    files: tuple[str, ...]
    size_gb: float | None
    model_type: str | None
    resolved_model_type: str | None
    arch_supported: bool | None
    template: TemplateShape
    error: str | None = None

    def verdict(self) -> tuple[str, list[str]]:
        """Return ("OK" | "WARN" | "BLOCKED", reasons)."""
        if self.error is not None:
            return "BLOCKED", [f"hub read failed: {self.error}"]
        blocked: list[str] = []
        warnings: list[str] = []
        missing = layout_missing(self.files)
        if missing:
            blocked.append(f"cache layout would skip it: missing {', '.join(missing)}")
        if self.arch_supported is False:
            blocked.append(
                f"no mlx-vlm package for model_type {self.model_type!r} "
                f"(resolves to {self.resolved_model_type!r})"
            )
        elif self.arch_supported is None:
            warnings.append(
                "architecture not checked"
                + (": no model_type in config.json" if self.model_type is None else "")
            )
        if self.template == "string-only":
            blocked.append(
                "text-only chat template (concatenates message content as a string); "
                "list-content messages fail at prefill"
            )
        elif self.template == "absent":
            warnings.append("no chat template found; the processor may supply one")
        elif self.template == "unknown":
            warnings.append("chat template shape not recognised; check it handles content parts")
        if blocked:
            return "BLOCKED", blocked + warnings
        return ("WARN", warnings) if warnings else ("OK", [])


ArchCheck = Callable[[object], tuple[str | None, str | None, bool | None]]
Fetcher = Callable[[str], HubCandidate]


def _default_arch_check() -> ArchCheck:
    """Use the harness's own resolution so hub and cache verdicts agree."""
    import os  # noqa: PLC0415 - deferred so --help never imports the monolith

    os.environ.setdefault("CHECK_MODELS_SKIP_IMPORT_PROBE", "1")
    from check_models import arch_precheck_for_model_type  # noqa: PLC0415 - see above

    return arch_precheck_for_model_type


def _fetch_text(url: str, headers: dict[str, str]) -> str | None:
    """Fetch one small hub file over HTTPS; None when it does not exist."""
    request = Request(url, headers=headers)  # noqa: S310 - https hub URL built by hf_hub_url
    try:
        with urlopen(request, timeout=_FETCH_TIMEOUT_S) as response:  # noqa: S310 - as above
            return response.read().decode("utf-8")
    except OSError:
        return None


def _template_from_tokenizer_config(text: str | None) -> str | None:
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    template = payload.get("chat_template") if isinstance(payload, dict) else None
    if isinstance(template, str):
        return template
    if isinstance(template, list):
        # transformers' named-template form: [{"name": ..., "template": ...}, ...]
        parts = [
            item.get("template")
            for item in template
            if isinstance(item, dict) and isinstance(item.get("template"), str)
        ]
        return "\n".join(parts) if parts else None
    return None


def fetch_candidate(repo_id: str, *, arch_check: ArchCheck | None = None) -> HubCandidate:
    """Read the file list, config.json and chat template of a hub repo."""
    from huggingface_hub import HfApi, hf_hub_url  # noqa: PLC0415 - network path only
    from huggingface_hub.utils import build_hf_headers  # noqa: PLC0415 - network path only

    check = arch_check if arch_check is not None else _default_arch_check()
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception as error:  # noqa: BLE001 - any hub failure is the verdict
        return HubCandidate(repo_id, (), None, None, None, None, "absent", error=str(error))
    files = tuple(sorted(sibling.rfilename for sibling in info.siblings or ()))
    sizes = [sibling.size for sibling in info.siblings or () if sibling.size]
    size_gb = round(sum(sizes) / 1e9, 1) if sizes else None
    headers = build_hf_headers()
    config_text = _fetch_text(hf_hub_url(repo_id, "config.json"), headers)
    model_type_raw: object = None
    if config_text is not None:
        try:
            config = json.loads(config_text)
        except json.JSONDecodeError:
            config = {}
        if isinstance(config, dict):
            model_type_raw = config.get("model_type") or config.get("speculators_model_type")
    model_type, resolved, supported = check(model_type_raw)
    template = _fetch_text(hf_hub_url(repo_id, "chat_template.jinja"), headers)
    if template is None:
        template = _template_from_tokenizer_config(
            _fetch_text(hf_hub_url(repo_id, "tokenizer_config.json"), headers)
        )
    return HubCandidate(
        repo_id=repo_id,
        files=files,
        size_gb=size_gb,
        model_type=model_type,
        resolved_model_type=resolved,
        arch_supported=supported,
        template=template_shape(template),
    )


def render(candidates: Sequence[HubCandidate]) -> str:
    """Render one line per candidate plus indented reasons."""
    lines: list[str] = []
    for candidate in candidates:
        status, reasons = candidate.verdict()
        size = f"{candidate.size_gb:.1f} GB" if candidate.size_gb is not None else "size ?"
        arch = candidate.resolved_model_type or candidate.model_type or "?"
        lines.append(
            f"{status:7} {size:>9}  {arch:24} template={candidate.template:17} {candidate.repo_id}"
        )
        lines.extend(f"          - {reason}" for reason in reasons)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, fetch: Fetcher = fetch_candidate) -> int:
    """CLI entry point; exit 1 when any candidate is blocked."""
    parser = argparse.ArgumentParser(
        prog="python -m tools.hub_precheck",
        description="Judge hub checkpoints before downloading: layout, architecture, template.",
    )
    parser.add_argument("repos", nargs="+", metavar="REPO", help="hub repo ids (org/name)")
    parser.add_argument("--json", action="store_true", help="emit one JSON object per line")
    args = parser.parse_args(argv)
    candidates = [fetch(repo_id) for repo_id in args.repos]
    if args.json:
        for candidate in candidates:
            status, reasons = candidate.verdict()
            print(json.dumps({**candidate.__dict__, "status": status, "reasons": reasons}))
    else:
        print(render(candidates))
    return 1 if any(candidate.verdict()[0] == "BLOCKED" for candidate in candidates) else 0


if __name__ == "__main__":
    raise SystemExit(main())
