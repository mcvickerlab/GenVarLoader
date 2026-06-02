from __future__ import annotations

import os
from hashlib import blake2b
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_extra_types.semantic_version import SemanticVersion

__all__ = ["FastaCache"]

FORMAT_VERSION = SemanticVersion.parse("1.0.0")
"""On-disk schema version for the .gvlfa cache."""
FINGERPRINT_WINDOW = 1 << 20  # 1 MiB
GVLFA_SUFFIX = ".gvlfa"
LEGACY_SUFFIX = ".gvl"
DATA_FILENAME = "sequence.bin"
METADATA_FILENAME = "metadata.json"


class SourceHints(BaseModel):
    filename: str
    absolute_path: str
    relative_path: str


class Fingerprint(BaseModel):
    algorithm: Literal["blake2b"] = "blake2b"
    n_bytes_hashed: int
    digest: str
    size_bytes: int


class FastaCache(BaseModel):
    format_version: SemanticVersion
    genvarloader_version: SemanticVersion
    contigs: dict[str, int]
    source: SourceHints
    fingerprint: Fingerprint


def _source_hints(source_fa: str | Path, gvlfa_dir: str | Path) -> SourceHints:
    source = Path(source_fa).resolve()
    dest = Path(gvlfa_dir).resolve()
    try:
        relative = os.path.relpath(source, dest)
    except ValueError:  # e.g. different drives on Windows
        relative = str(source)
    return SourceHints(
        filename=source.name,
        absolute_path=str(source),
        relative_path=str(relative),
    )


def resolve_source(gvlfa_dir: str | Path, meta: FastaCache) -> Path | None:
    """Locate the source FASTA: sibling, then relative, then absolute. First hit wins."""
    gvlfa_dir = Path(gvlfa_dir)
    candidates = [
        gvlfa_dir.parent / meta.source.filename,
        gvlfa_dir / meta.source.relative_path,
        Path(meta.source.absolute_path),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def fingerprint(path: str | Path) -> Fingerprint:
    """Cheap content fingerprint: blake2b of the first FINGERPRINT_WINDOW bytes,
    plus the total file size (catches changes past the hashed window)."""
    path = Path(path)
    size = path.stat().st_size
    n = min(FINGERPRINT_WINDOW, size)
    h = blake2b()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return Fingerprint(n_bytes_hashed=n, digest=h.hexdigest(), size_bytes=size)
