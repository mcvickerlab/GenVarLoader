from __future__ import annotations

import os
from hashlib import blake2b
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import pysam
from pydantic import BaseModel
from pydantic_extra_types.semantic_version import SemanticVersion
from tqdm.auto import tqdm

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


def _gvl_version() -> SemanticVersion:
    return SemanticVersion.parse(version("genvarloader"))


def _contig_lengths(source_fa: str | Path) -> dict[str, int]:
    with pysam.FastaFile(str(source_fa)) as f:
        return {c: f.get_reference_length(c) for c in f.references}


def _write_sequence(source_fa: Path, gvlfa_dir: Path, contigs: dict[str, int]) -> None:
    total = sum(contigs.values())
    data = np.memmap(gvlfa_dir / DATA_FILENAME, dtype=np.uint8, mode="w+", shape=total)
    offset = 0
    with pysam.FastaFile(str(source_fa)) as f:
        pbar = tqdm(total=total, unit=" nucleotide")
        for c in contigs:
            c_seq = np.frombuffer(f.fetch(c).encode("ascii").upper(), "S1")
            data[offset : offset + len(c_seq)] = c_seq.view(np.uint8)
            offset += len(c_seq)
            pbar.update(len(c_seq))
        pbar.close()
    data.flush()


def build(source_fa: str | Path, gvlfa_dir: str | Path) -> FastaCache:
    """Build a fresh .gvlfa cache containing all contigs of the source FASTA."""
    source_fa = Path(source_fa)
    gvlfa_dir = Path(gvlfa_dir)
    gvlfa_dir.mkdir(parents=True, exist_ok=True)
    contigs = _contig_lengths(source_fa)
    _write_sequence(source_fa, gvlfa_dir, contigs)
    meta = FastaCache(
        format_version=FORMAT_VERSION,
        genvarloader_version=_gvl_version(),
        contigs=contigs,
        source=_source_hints(source_fa, gvlfa_dir),
        fingerprint=fingerprint(source_fa),
    )
    (gvlfa_dir / METADATA_FILENAME).write_text(meta.model_dump_json())
    return meta


def _check_format_version(meta: FastaCache, gvlfa_dir: Path) -> None:
    if meta.format_version.major > FORMAT_VERSION.major:
        raise ValueError(
            f"FASTA cache at {gvlfa_dir} has format version {meta.format_version}, "
            f"newer than supported {FORMAT_VERSION}. Upgrade genvarloader."
        )


def _data_size_ok(gvlfa_dir: Path, meta: FastaCache) -> bool:
    data_path = Path(gvlfa_dir) / DATA_FILENAME
    if not data_path.exists():
        return False
    return data_path.stat().st_size == sum(meta.contigs.values())


def _fingerprints_match(stored: Fingerprint, source_fa: Path) -> bool:
    if stored.size_bytes != Path(source_fa).stat().st_size:
        return False
    return fingerprint(source_fa).digest == stored.digest


def load(gvlfa_dir: str | Path) -> tuple[FastaCache, Path | None, Literal["fresh", "stale", "unvalidated"]]:
    """Read cache metadata and classify it: 'fresh' | 'stale' | 'unvalidated'.

    Raises ValueError if the format version is too new to read.
    """
    gvlfa_dir = Path(gvlfa_dir)
    meta = FastaCache.model_validate_json((gvlfa_dir / METADATA_FILENAME).read_text())
    _check_format_version(meta, gvlfa_dir)
    source = resolve_source(gvlfa_dir, meta)
    if source is None:
        status = "unvalidated"
    elif _fingerprints_match(meta.fingerprint, source):
        status = "fresh"
    else:
        status = "stale"
    return meta, source, status
