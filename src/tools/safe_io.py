"""Symlink-resistant file helpers for maintenance tools."""

from __future__ import annotations

import os
import stat
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

MAX_SAFE_TEXT_FILE_BYTES: Final[int] = 16 * 1024 * 1024
SAFE_TEXT_FILE_READ_CHUNK_BYTES: Final[int] = 64 * 1024


def _reject_symlink_parent(path: Path) -> None:
    """Reject symlinked parent directories before tool file writes."""
    for parent in (path.parent, *path.parent.parents):
        if parent.is_symlink():
            msg = f"Refusing to access file through symlinked directory: {parent}"
            raise OSError(msg)


def _canonical_file_path(path: Path) -> Path:
    """Resolve existing parent links without following the final file path."""
    return path.parent.resolve(strict=False) / path.name


def read_text_no_follow(
    path: Path,
    *,
    max_bytes: int = MAX_SAFE_TEXT_FILE_BYTES,
    encoding: str = "utf-8",
) -> str:
    """Read text from a regular file without following symlinks."""
    _reject_symlink_parent(path)
    path = _canonical_file_path(path)
    _reject_symlink_parent(path)
    if path.is_symlink():
        msg = f"Refusing to follow symlink: {path}"
        raise OSError(msg)

    fd = os.open(  # skylos: ignore[SKY-D215] path is canonicalized and opened with O_NOFOLLOW.
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
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
    if file_stat.st_size > max_bytes:
        os.close(fd)
        msg = f"Refusing to read {path}: file size {file_stat.st_size} exceeds {max_bytes} bytes"
        raise OSError(msg)

    chunks: list[bytes] = []
    remaining = max_bytes + 1
    try:
        while remaining > 0:
            chunk = os.read(  # skylos: ignore[SKY-P401] bounded by max_bytes.
                fd,
                min(remaining, SAFE_TEXT_FILE_READ_CHUNK_BYTES),
            )
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
    return data.decode(encoding)


def write_text_no_follow(path: Path, content: str, *, mode: int = 0o666) -> None:
    """Write UTF-8 text to a regular file without following symlinks."""
    _reject_symlink_parent(path)
    path = _canonical_file_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_parent(path)
    if path.is_symlink():
        msg = f"Refusing to follow symlink: {path}"
        raise OSError(msg)

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(  # skylos: ignore[SKY-D215] path is canonicalized and opened with O_NOFOLLOW.
        path,
        flags,
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

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
