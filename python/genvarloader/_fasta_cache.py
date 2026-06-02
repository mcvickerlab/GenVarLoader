from __future__ import annotations

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
