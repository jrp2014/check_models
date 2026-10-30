"""Pre-download check for Hugging Face hub candidates.

Before spending tens of gigabytes on a checkpoint, judge it from a few small
hub reads the way ``check_models`` will judge it once cached:

1. **Layout**: the server-style file rule (``config.json``,
   ``tokenizer_config.json``, safetensors weights).
2. **Architecture**: the ``model_type`` resolves (via ``MODEL_REMAPPING``) to
   an installed ``mlx_vlm/models/`` package whose ``__init__`` binds ``Model``
   and its config class (read statically, never imported).
3. **Chat template shape**: whether the template iterates message content
   parts (multimodal) or concatenates content as a string (text-only), which
   fails at prefill for vision families that send list-content messages.
4. **Memory**: the safetensors weights mlx-vlm would load (the shards the
   ``model.safetensors.index.json`` names, else the root-level
   ``*.safetensors``) against this Mac's unified memory (blocked when larger:
   they cannot be held) and Metal's recommended working set (a warning when
   larger: expect paging or a load failure). Runtime allocations come on
   top, so fitting weights is necessary, not sufficient.
5. **Lineage** (facts, never a verdict): the ``base_model`` chain the model
   cards declare, up to ``_LINEAGE_MAX_DEPTH`` hops, and the cached models
   that are, or declare, one of those bases. A shared base marks a variant of
   something already cached (a re-quantisation or fine-tune), worth reading
   the card before downloading. It depends on publishers declaring bases.

Only ``config.json``, the chat template, the safetensors index and the
model-card metadata of the candidate and its declared bases are fetched,
over HTTP(S); the local Hugging Face cache is only read, for the cached
models' own declared bases. Verdicts are hints, never proof.

Usage::

    python -m tools.hub_precheck mlx-community/Some-VLM-4bit [more repos...]

Exit status 1 when any candidate is blocked.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

_LAYOUT_REQUIRED: Final[tuple[str, ...]] = ("config.json", "tokenizer_config.json")
_FETCH_TIMEOUT_S: Final[int] = 30
# Hops of declared base_model followed from a candidate: an MLX conversion of
# a fine-tune of a base model is three.
_LINEAGE_MAX_DEPTH: Final[int] = 3

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


def selected_weight_files(files: Iterable[str], index_text: str | None) -> tuple[str, ...]:
    """The safetensors files mlx-vlm's loader would read, by its static rules.

    Mirrors ``mlx_vlm.utils.load_model``: the shards named by the index's
    ``weight_map`` that exist, else every root-level ``*.safetensors`` except
    ``consolidated.safetensors``. Nested files (alternate checkpoints such as
    ``original/``) are never loaded, so they never count.
    """
    names = set(files)
    if index_text is not None:
        try:
            payload = json.loads(index_text)
        except json.JSONDecodeError:
            payload = None
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if isinstance(weight_map, dict):
            shards = sorted({str(shard) for shard in weight_map.values() if shard} & names)
            if shards:
                return tuple(shards)
    return tuple(
        sorted(
            name
            for name in names
            if "/" not in name
            and name.endswith(".safetensors")
            and not name.endswith("consolidated.safetensors")
        )
    )


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
    weights_gb: float | None = None
    memory_gb: float | None = None
    working_set_gb: float | None = None
    # The hub listed no size for at least one selected weight file, so the
    # memory fit is unknown rather than assumed.
    weights_size_unknown: bool = False
    # Which file the chat template was read from (None when none was found).
    template_source: str | None = None
    # The hub's task label (e.g. "image-text-to-text"); recorded, not gated on.
    pipeline_tag: str | None = None
    # config.json names checkpoint-shipped model code (mlx-vlm loads it as "custom").
    model_file: str | None = None
    # The declared base_model chain, nearest first (empty when none is declared).
    base_models: tuple[str, ...] = ()
    # (cached repo, the shared id): cached models that are, or declare, one of
    # this candidate's bases, or that declare this candidate as their base.
    cached_relatives: tuple[tuple[str, str], ...] = ()
    already_cached: bool = False
    # Bases whose own card could not be read, so the chain may stop early.
    lineage_unreadable: tuple[str, ...] = ()

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
        elif self.model_file is not None:
            warnings.append(
                f"config.json declares model_file={self.model_file!r}: mlx-vlm imports and "
                "runs that file from the checkpoint instead of its own model packages"
            )
        elif self.arch_supported is None:
            warnings.append(
                "architecture not checked"
                + (": no model_type in config.json" if self.model_type is None else "")
            )
        if self.weights_gb is not None and self.memory_gb and self.weights_gb > self.memory_gb:
            blocked.append(
                f"weights ({self.weights_gb:.1f} GB) exceed this Mac's "
                f"{self.memory_gb:.1f} GB of unified memory"
            )
        elif (
            self.weights_gb is not None
            and self.working_set_gb
            and self.weights_gb > self.working_set_gb
        ):
            warnings.append(
                f"weights ({self.weights_gb:.1f} GB) exceed Metal's recommended working set "
                f"({self.working_set_gb:.1f} GB); expect paging or a load failure"
            )
        if self.weights_size_unknown:
            warnings.append(
                "the hub lists no size for some weight files the loader would read; "
                "memory fit not assessed"
            )
        source = f" in {self.template_source}" if self.template_source else ""
        if self.template == "string-only":
            blocked.append(
                f"text-only chat template{source} (concatenates message content as a string); "
                "list-content messages fail at prefill"
            )
        elif self.template == "absent":
            warnings.append(
                "no chat template found in chat_template.jinja/.json, processor_config.json "
                "or tokenizer_config.json; the processor may supply one"
            )
        elif self.template == "unknown":
            warnings.append(
                f"chat template shape{source} not recognised; check it handles content parts"
            )
        if blocked:
            return "BLOCKED", blocked + warnings
        return ("WARN", warnings) if warnings else ("OK", [])


ArchCheck = Callable[[object], tuple[str | None, str | None, bool | None]]
MemoryCheck = Callable[[], tuple[int | None, int | None]]
Fetcher = Callable[[str], HubCandidate]
# A repo id's declared bases, and every cached model's declared bases.
BasesLookup = Callable[[str], tuple[str, ...]]
CachedBases = Callable[[], Mapping[str, tuple[str, ...]]]


def declared_bases(card_data: object) -> tuple[str, ...]:
    """A model card's ``base_model``, which may be one id or a list, as a tuple."""
    getter = getattr(card_data, "get", None)
    raw = getter("base_model") if callable(getter) else None
    items = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    return tuple(
        dict.fromkeys(item.strip() for item in items if isinstance(item, str) and item.strip())
    )


class BaseCardUnreadableError(LookupError):
    """A declared base's model card could not be read from the hub."""


def lineage(first: Sequence[str], bases_of: BasesLookup) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The declared base chain from a candidate's own bases, and the bases left unread.

    Breadth-first, without repeats, for at most ``_LINEAGE_MAX_DEPTH`` hops,
    looking up no further than it reports; repo ids compare
    case-insensitively, as the hub resolves them. A base whose card cannot be
    read is kept in the chain and also returned as unread, since the chain
    may continue past it.
    """
    chain: list[str] = []
    unreadable: list[str] = []
    seen: set[str] = set()
    frontier = list(first)
    for hop in range(_LINEAGE_MAX_DEPTH):
        following: list[str] = []
        for repo in frontier:
            if repo.lower() in seen:
                continue
            seen.add(repo.lower())
            chain.append(repo)
            if hop < _LINEAGE_MAX_DEPTH - 1:
                try:
                    following.extend(bases_of(repo))
                except BaseCardUnreadableError:
                    unreadable.append(repo)
        frontier = following
        if not frontier:
            break
    return tuple(chain), tuple(unreadable)


def cached_relatives(
    repo_id: str, chain: Sequence[str], cached: Mapping[str, tuple[str, ...]]
) -> tuple[tuple[str, str], ...]:
    """(cached repo, shared id) for cached models related to the candidate by declared bases.

    Related means the cached model is one of the candidate's bases, declares
    one of them, or declares the candidate itself as its base.
    """
    own = repo_id.lower()
    lineage_ids = {own, *(base.lower() for base in chain)}
    related: list[tuple[str, str]] = []
    for cached_id, bases in sorted(cached.items()):
        if cached_id.lower() == own:
            continue
        shared = next((rid for rid in (cached_id, *bases) if rid.lower() in lineage_ids), None)
        if shared is not None:
            related.append((cached_id, shared))
    return tuple(related)


def _hub_declared_bases(repo_id: str) -> tuple[str, ...]:
    """One hub repo's declared bases; raises BaseCardUnreadableError when unreadable."""
    from huggingface_hub import HfApi  # noqa: PLC0415 - network path only

    try:
        info = HfApi().model_info(repo_id)
    except Exception as error:
        # Any hub failure leaves this base unread; lineage() records it.
        raise BaseCardUnreadableError(repo_id) from error
    return declared_bases(getattr(info, "card_data", None))


@functools.cache
def _cached_declared_bases() -> Mapping[str, tuple[str, ...]]:
    """Every cached model's declared bases, from the README in its main snapshot."""
    from huggingface_hub import metadata_load, scan_cache_dir  # noqa: PLC0415 - cache path only
    from huggingface_hub.errors import CacheNotFound  # noqa: PLC0415 - cache path only

    try:
        cache = scan_cache_dir()
    except (CacheNotFound, OSError):
        # No readable cache: nothing cached to relate a candidate to.
        return {}
    bases_by_repo: dict[str, tuple[str, ...]] = {}
    for repo in cache.repos:
        if repo.repo_type != "model":
            continue
        # The main ref's snapshot is the one check_models loads.
        revisions = sorted(repo.revisions, key=lambda revision: "main" not in revision.refs)
        card = revisions[0].snapshot_path / "README.md" if revisions else None
        bases: tuple[str, ...] = ()
        if card is not None and card.is_file():
            try:
                bases = declared_bases(metadata_load(card))
            except (OSError, ValueError, yaml.YAMLError):
                # An unreadable or malformed card declares no base.
                bases = ()
        bases_by_repo[repo.repo_id] = bases
    return bases_by_repo


def _default_arch_check() -> ArchCheck:
    """Use the harness's own resolution so hub and cache verdicts agree."""
    # Deferred so --help never imports the monolith.
    os.environ.setdefault("CHECK_MODELS_SKIP_IMPORT_PROBE", "1")
    from check_models import arch_precheck_for_model_type  # noqa: PLC0415 - see above

    return arch_precheck_for_model_type


def _default_memory_check() -> tuple[int | None, int | None]:
    """(unified memory bytes, Metal recommended working set bytes) from the harness."""
    # Deferred so --help never imports the monolith.
    os.environ.setdefault("CHECK_MODELS_SKIP_IMPORT_PROBE", "1")
    from check_models import (  # noqa: PLC0415 - see above
        _get_recommended_working_set_bytes,
        _get_total_memory_bytes,
    )

    return _get_total_memory_bytes(), _get_recommended_working_set_bytes()


def _gigabytes(value: int | None) -> float | None:
    return round(value / 1e9, 1) if value else None


def _fetch_text(url: str, headers: dict[str, str]) -> str | None:
    """Fetch one small HTTP(S) hub file; None when unsupported or unavailable."""
    if urlparse(url).scheme not in {"http", "https"}:
        return None
    request = Request(url, headers=headers)  # noqa: S310 - HTTP(S) scheme checked above
    try:
        with urlopen(request, timeout=_FETCH_TIMEOUT_S) as response:  # noqa: S310 - as above
            return response.read().decode("utf-8")
    except OSError:
        return None


# The template the processor ends up with, in transformers'
# ProcessorMixin.get_processor_dict precedence: a "chat_template" key in
# processor_config.json overrides the files; the legacy chat_template.json
# wins over chat_template.jinja; the tokenizer's config applies only when the
# processor has none. Reading only .jinja and tokenizer_config misjudged
# checkpoints whose multimodal template lives in chat_template.json (Idefics3
# looked text-only; pixtral looked template-less). Kept equal to
# check_models._CHAT_TEMPLATE_FILES (a test checks both against the loader).
_TEMPLATE_SOURCES: Final[tuple[str, ...]] = (
    "processor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "tokenizer_config.json",
)


def _template_from_tokenizer_config(text: str | None) -> str | None:
    """Read ``chat_template`` from a JSON config (string or named-template list)."""
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


type TextFetcher = Callable[[str], str | None]


def _config_facts(fetch: TextFetcher) -> tuple[object, str | None]:
    """config.json's raw model_type and its declared ``model_file``, when readable."""
    config_text = fetch("config.json")
    if config_text is None:
        return None, None
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(config, dict):
        return None, None
    declared_file = config.get("model_file")
    model_file = declared_file if isinstance(declared_file, str) and declared_file else None
    return config.get("model_type") or config.get("speculators_model_type"), model_file


def _first_template(files: Iterable[str], fetch: TextFetcher) -> tuple[str | None, str | None]:
    """The chat template AutoProcessor would read, and the file it came from."""
    names = set(files)
    for source in _TEMPLATE_SOURCES:
        if source not in names:
            continue
        text = fetch(source)
        found = text if source.endswith(".jinja") else _template_from_tokenizer_config(text)
        if found:
            return found, source
    return None, None


def _selected_weight_sizes(
    files: Sequence[str], size_by_name: dict[str, int | None], fetch: TextFetcher
) -> list[int | None]:
    """Hub sizes of the weight files the loader would read (None where unlisted)."""
    index_text = (
        fetch("model.safetensors.index.json")
        if "model.safetensors.index.json" in size_by_name
        else None
    )
    return [size_by_name.get(name) for name in selected_weight_files(files, index_text)]


def fetch_candidate(
    repo_id: str,
    *,
    arch_check: ArchCheck | None = None,
    memory_check: MemoryCheck | None = None,
    bases_of: BasesLookup | None = None,
    cached_bases: CachedBases | None = None,
) -> HubCandidate:
    """Read the file list, config.json, chat template and lineage of a hub repo."""
    from huggingface_hub import HfApi, hf_hub_url  # noqa: PLC0415 - network path only
    from huggingface_hub.utils import build_hf_headers  # noqa: PLC0415 - network path only

    check = arch_check if arch_check is not None else _default_arch_check()
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception as error:  # noqa: BLE001 - any hub failure is the verdict
        return HubCandidate(repo_id, (), None, None, None, None, "absent", error=str(error))
    files = tuple(sorted(sibling.rfilename for sibling in info.siblings or ()))
    # The whole repository's size is shown for information only; the memory
    # verdict uses the weights the loader would select.
    sizes = [sibling.size for sibling in info.siblings or () if sibling.size]
    size_by_name = {sibling.rfilename: sibling.size for sibling in info.siblings or ()}
    memory_bytes, working_set_bytes = (memory_check or _default_memory_check)()
    headers = build_hf_headers()

    def fetch(file_name: str) -> str | None:
        return _fetch_text(hf_hub_url(repo_id, file_name), headers)

    weight_sizes = _selected_weight_sizes(files, size_by_name, fetch)
    weights_known = bool(weight_sizes) and all(weight_sizes)
    model_type_raw, model_file = _config_facts(fetch)
    model_type, resolved, supported = check(model_type_raw)
    # mlx-vlm imports a declared model_file from the checkpoint instead of its
    # own packages (get_model_and_args), so the package check does not apply.
    template, template_source = _first_template(files, fetch)
    chain, unreadable = lineage(
        declared_bases(getattr(info, "card_data", None)), bases_of or _hub_declared_bases
    )
    cached = (cached_bases or _cached_declared_bases)()
    return HubCandidate(
        base_models=chain,
        cached_relatives=cached_relatives(repo_id, chain, cached),
        already_cached=repo_id.lower() in {cached_id.lower() for cached_id in cached},
        lineage_unreadable=unreadable,
        repo_id=repo_id,
        files=files,
        size_gb=round(sum(sizes) / 1e9, 1) if sizes else None,
        model_type=model_type,
        resolved_model_type=resolved,
        arch_supported=None if model_file is not None else supported,
        template=template_shape(template),
        template_source=template_source,
        pipeline_tag=getattr(info, "pipeline_tag", None),
        model_file=model_file,
        weights_gb=_gigabytes(sum(size or 0 for size in weight_sizes)) if weights_known else None,
        memory_gb=_gigabytes(memory_bytes),
        working_set_gb=_gigabytes(working_set_bytes),
        weights_size_unknown=bool(weight_sizes) and not weights_known,
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
        facts = [
            f"{label}: {value}"
            for label, value in (
                ("template from", candidate.template_source),
                ("pipeline_tag", candidate.pipeline_tag),
                ("base", " → ".join(candidate.base_models)),
                ("already cached", "yes" if candidate.already_cached else ""),
                ("card unreadable, chain may continue", ", ".join(candidate.lineage_unreadable)),
            )
            if value
        ]
        if facts:
            lines.append("          (" + "; ".join(facts) + ")")
        if candidate.cached_relatives:
            related = ", ".join(
                cached_id if shared == cached_id else f"{cached_id} (via {shared})"
                for cached_id, shared in candidate.cached_relatives
            )
            lines.append(f"          (cached models with a shared base: {related})")
        lines.extend(f"          - {reason}" for reason in reasons)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, fetch: Fetcher = fetch_candidate) -> int:
    """CLI entry point; exit 1 when any candidate is blocked."""
    parser = argparse.ArgumentParser(
        prog="python -m tools.hub_precheck",
        description=(
            "Judge hub checkpoints before downloading: layout, architecture, template, "
            "memory and lineage."
        ),
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
