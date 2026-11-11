"""Symlink-safe file helpers shared by ``check_models.py`` and the maintenance tools.

Standard library only, and shipped in the wheel, so the installed script and
the checkout's tools run the same code.

The contract, for every helper:

* **Parent directories** are never reached through a symlink. The only
  exceptions are the macOS root aliases in ``ALLOWED_SYSTEM_SYMLINK_PARENTS``
  (``/tmp`` and ``/var``), and only while each still points at its
  ``/private`` target.
* **The final path component** is never followed. A symlink there is refused,
  as is anything that is not a regular file.
* **Refusals and failures** are ``OSError`` (a ``UnicodeError`` for text that
  cannot be encoded or decoded).

Per helper:

* ``read_text_no_follow`` returns the file's UTF-8 text and refuses a file
  larger than ``max_bytes``.
* ``write_text_no_follow`` / ``write_bytes_no_follow`` create missing parent
  directories, then create the file or truncate it (``append=True`` adds to
  the end instead). A new file gets ``mode`` less the umask; an existing file
  keeps its permission bits. A failed write can leave a partial file.
* ``replace_text_no_follow`` is the all-or-nothing write: the text goes to a
  temporary file beside the target, is flushed to disk, and is renamed over
  the target. Until the rename the previous file is untouched, and a failure
  removes the temporary file. A missing target is created (``0o666`` less the
  umask); an existing one keeps its permission bits.
* ``open_regular_file_no_follow`` applies the path rules and returns a file
  descriptor, for callers that read or write in their own way.
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path
from typing import Final

MAX_SAFE_TEXT_FILE_BYTES: Final[int] = 16 * 1024 * 1024
SAFE_TEXT_FILE_READ_CHUNK_BYTES: Final[int] = 64 * 1024
_ROOT: Final[Path] = Path(os.sep)
ALLOWED_SYSTEM_SYMLINK_PARENTS: Final[dict[Path, Path]] = {
    _ROOT / "tmp": _ROOT / "private" / "tmp",
    _ROOT / "var": _ROOT / "private" / "var",
}


def _is_allowed_system_symlink_parent(path: Path) -> bool:
    """Return true for a trusted macOS root alias that still has its expected target."""
    expected_target = ALLOWED_SYSTEM_SYMLINK_PARENTS.get(path)
    if expected_target is None:
        return False
    try:
        return path.resolve(strict=True) == expected_target
    except OSError:
        return False


def _reject_symlink_parent(path: Path) -> None:
    """Refuse a path whose parent chain contains an untrusted symlink."""
    for parent in (path.parent, *path.parent.parents):
        if parent.is_symlink() and not _is_allowed_system_symlink_parent(parent):
            msg = f"Refusing to access file through symlinked directory: {parent}"
            raise OSError(msg)


def _checked_path(path: Path, *, create_parents: bool = False) -> Path:
    """Return ``path`` with its parents resolved, after the parent-symlink checks.

    The parents are checked as given and again once resolved, so neither a
    relative path nor a trusted root alias can hide a symlinked directory.
    """
    _reject_symlink_parent(path)
    canonical = path.parent.resolve(strict=False) / path.name
    if create_parents:
        canonical.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_parent(canonical)
    return canonical


def _open_checked(path: Path, flags: int, mode: int) -> int:
    """Open an already-checked path as a regular file, never through a final symlink."""
    if path.is_symlink():
        msg = f"Refusing to follow symlink: {path}"
        raise OSError(msg)
    fd = os.open(  # skylos: ignore[SKY-D215] parents are checked and the open is O_NOFOLLOW.
        path,
        flags | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        file_stat = os.fstat(fd)
    except OSError:
        os.close(fd)
        raise
    if not stat.S_ISREG(file_stat.st_mode):
        os.close(fd)
        msg = f"Refusing to access non-regular file: {path}"
        raise OSError(msg)
    return fd


def open_regular_file_no_follow(path: Path, flags: int, mode: int = 0o666) -> int:
    """Return a descriptor for a regular file, opened under the module's path rules."""
    return _open_checked(_checked_path(path), flags, mode)


def read_text_no_follow(path: Path, *, max_bytes: int = MAX_SAFE_TEXT_FILE_BYTES) -> str:
    """Read UTF-8 text from a regular file of at most ``max_bytes`` bytes."""
    path = _checked_path(path)
    fd = _open_checked(path, os.O_RDONLY, 0o666)
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    try:
        file_size = os.fstat(fd).st_size
        if file_size > max_bytes:
            msg = f"Refusing to read {path}: file size {file_size} exceeds {max_bytes} bytes"
            raise OSError(msg)
        # The size check is not trusted alone: the file can grow while it is read.
        while remaining > 0:
            chunk = os.read(fd, min(remaining, SAFE_TEXT_FILE_READ_CHUNK_BYTES))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    data = b"".join(chunks)
    if len(data) > max_bytes:
        msg = f"Refusing to read {path}: content exceeds {max_bytes} bytes"
        raise OSError(msg)
    return data.decode("utf-8")


def write_bytes_no_follow(
    path: Path,
    content: bytes,
    *,
    append: bool = False,
    mode: int = 0o666,
) -> None:
    """Write bytes to a regular file, creating it and its parent directories as needed."""
    path = _checked_path(path, create_parents=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    with os.fdopen(_open_checked(path, flags, mode), "ab") as handle:
        handle.write(content)


def write_text_no_follow(
    path: Path,
    content: str,
    *,
    append: bool = False,
    mode: int = 0o666,
) -> None:
    """Write UTF-8 text to a regular file, creating it and its parent directories as needed."""
    write_bytes_no_follow(path, content.encode("utf-8"), append=append, mode=mode)


def replace_text_no_follow(path: Path, content: str) -> None:
    """Replace a file's whole text, or create it, through a sibling file and a rename.

    A failure before the rename (encoding, disk full, interrupt) leaves the
    previous file untouched rather than truncated, and removes the sibling.
    """
    path = _checked_path(path, create_parents=True)
    data = content.encode("utf-8")
    try:
        original_mode: int | None = os.lstat(path).st_mode
    except FileNotFoundError:
        original_mode = None
    if original_mode is not None and not stat.S_ISREG(original_mode):
        msg = f"Refusing to replace non-regular file: {path}"
        raise OSError(msg)

    staging = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = _open_checked(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as handle:
            if original_mode is not None:
                os.fchmod(handle.fileno(), stat.S_IMODE(original_mode))
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        staging.replace(path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
