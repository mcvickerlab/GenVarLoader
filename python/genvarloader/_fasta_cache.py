"""FASTA cache (``.gvlfa``) build and validation logic.

The cache is built atomically (temp directory + :func:`os.replace`) under a best-effort
``filelock``, so concurrent builders sharing one reference FASTA are safe. The cache
auto-rebuilds from its source when stale or missing (source fingerprint mismatch or
incomplete on-disk data).
"""

from __future__ import annotations

import os
import shutil
import warnings
from hashlib import blake2b
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import pysam
from filelock import FileLock
from loguru import logger
from pydantic import BaseModel
from pydantic_extra_types.semantic_version import SemanticVersion
from tqdm.auto import tqdm

from ._atomic import SkipPublish, atomic_dir

__all__ = ["FastaCache", "ensure_cache"]

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


def _build_into(source_fa: Path, target_dir: Path, dest_for_hints: Path) -> FastaCache:
    """Write sequence.bin + metadata.json into `target_dir`.

    `dest_for_hints` is the *final* cache dir (a sibling of `target_dir` at the
    same depth) used to compute source path hints, so the stored relative path is
    correct after publish.
    """
    contigs = _contig_lengths(source_fa)
    _write_sequence(source_fa, target_dir, contigs)
    meta = FastaCache(
        format_version=FORMAT_VERSION,
        genvarloader_version=_gvl_version(),
        contigs=contigs,
        source=_source_hints(source_fa, dest_for_hints),
        fingerprint=fingerprint(source_fa),
    )
    (target_dir / METADATA_FILENAME).write_text(meta.model_dump_json())
    return meta


def build(source_fa: str | Path, gvlfa_dir: str | Path) -> FastaCache:
    """Build a fresh .gvlfa cache containing all contigs of the source FASTA.

    Builds into a private sibling temp dir and atomically publishes it, so a
    concurrent builder or an interrupted build never leaves a partial cache.
    """
    source_fa = Path(source_fa)
    gvlfa_dir = Path(gvlfa_dir)
    meta_holder: dict[str, FastaCache] = {}
    with atomic_dir(gvlfa_dir, overwrite=True) as tmp:
        meta_holder["meta"] = _build_into(source_fa, tmp, gvlfa_dir)
    return meta_holder["meta"]


def _ensure_built(source_fa: Path, gvlfa_dir: Path) -> FastaCache:
    """Build the cache under a best-effort lock, double-checking inside the lock
    that another job hasn't already published a fresh cache."""
    with atomic_dir(gvlfa_dir, overwrite=True) as tmp:
        if gvlfa_dir.exists():
            try:
                meta, _source, status = load(gvlfa_dir)
                if status == "fresh" and _data_size_ok(gvlfa_dir, meta):
                    raise SkipPublish
            except SkipPublish:
                raise
            except Exception:
                pass  # unreadable/corrupt -> fall through and rebuild
        _build_into(source_fa, tmp, gvlfa_dir)
    return FastaCache.model_validate_json((gvlfa_dir / METADATA_FILENAME).read_text())


def _check_format_version(meta: FastaCache, gvlfa_dir: Path) -> None:
    if meta.format_version.major > FORMAT_VERSION.major:
        raise ValueError(
            f"FASTA cache at {gvlfa_dir} has format version {meta.format_version}, "
            f"newer than supported {FORMAT_VERSION}. Upgrade genvarloader."
        )


def _data_size_ok(gvlfa_dir: Path, meta: FastaCache) -> bool:
    # NOTE: only total byte count is checked here. A same-size corruption of
    # sequence.bin is not detected; full-content hashing would be prohibitively
    # expensive for large reference genomes.
    data_path = Path(gvlfa_dir) / DATA_FILENAME
    if not data_path.exists():
        return False
    return data_path.stat().st_size == sum(meta.contigs.values())


def _fingerprints_match(stored: Fingerprint, source_fa: Path) -> bool:
    if stored.size_bytes != Path(source_fa).stat().st_size:
        return False
    return fingerprint(source_fa).digest == stored.digest


def _matching_published_cache(
    gvlfa_dir: Path, contigs: dict[str, int], fp: Fingerprint
) -> FastaCache | None:
    try:
        meta = FastaCache.model_validate_json(
            (gvlfa_dir / METADATA_FILENAME).read_text()
        )
    except (OSError, ValueError):
        return None
    _check_format_version(meta, gvlfa_dir)
    if (
        meta.contigs != contigs
        or meta.fingerprint != fp
        or not _data_size_ok(gvlfa_dir, meta)
    ):
        return None
    return meta


def _same_source_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        os.path.samestat(before, after)
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    )


def _same_cleanup_identity(before: os.stat_result, after: os.stat_result) -> bool:
    # Renaming changes ctime on POSIX, so cleanup compares only inode identity and
    # content-relevant metadata that remain stable across the private-name claim.
    return (
        os.path.samestat(before, after)
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )


def _restore_cleanup_candidate(candidate: Path, legacy_gvl: Path) -> None:
    """Restore a displaced replacement without overwriting a newer pathname."""
    try:
        os.link(candidate, legacy_gvl, follow_symlinks=False)
    except FileExistsError:
        logger.warning(
            f"Could not restore replaced legacy cache to {legacy_gvl} because a "
            f"new file appeared there; preserving it at {candidate}."
        )
        return
    except OSError as exc:
        if os.name == "nt":
            try:
                os.rename(candidate, legacy_gvl)
            except OSError as rename_exc:
                logger.warning(
                    f"Could not restore replaced legacy cache to {legacy_gvl}; "
                    f"preserving it at {candidate}: {rename_exc!r}"
                )
            return
        logger.warning(
            f"Could not restore replaced legacy cache to {legacy_gvl}; "
            f"preserving it at {candidate}: {exc!r}"
        )
        return
    try:
        candidate.unlink()
    except OSError as exc:
        logger.warning(
            f"Could not remove restored cleanup candidate {candidate}; both it and "
            f"the public legacy cache {legacy_gvl} remain: {exc!r}"
        )


def _cleanup_legacy_source(legacy_gvl: Path, source_identity: os.stat_result) -> None:
    candidate = legacy_gvl.with_name(
        f"{legacy_gvl.name}.cleanup.{uuid4().hex[:8]}"
    )
    try:
        os.replace(legacy_gvl, candidate)
    except OSError as exc:
        logger.warning(
            f"Could not claim legacy cache for cleanup at {legacy_gvl}; leaving "
            f"post-publication cleanup unresolved: {exc!r}"
        )
        return

    try:
        candidate_identity = candidate.stat()
    except OSError as exc:
        logger.warning(
            f"Could not inspect cleanup candidate {candidate}; preserving it after "
            f"destination publication: {exc!r}"
        )
        return
    if _same_cleanup_identity(source_identity, candidate_identity):
        try:
            candidate.unlink()
        except OSError as exc:
            logger.warning(
                f"Could not remove cleanup candidate {candidate}; preserving it "
                f"after destination publication: {exc!r}"
            )
        return

    logger.info(
        f"Legacy cache {legacy_gvl} was replaced during migration; restoring the "
        "replacement."
    )
    _restore_cleanup_candidate(candidate, legacy_gvl)


def migrate_legacy(
    source_fa: str | Path, legacy_gvl: str | Path, gvlfa_dir: str | Path
) -> FastaCache:
    """Upgrade a legacy flat .gvl cache to a .gvlfa dir by reusing its bytes.

    Reads contig lengths and fingerprints the source *before* touching the legacy
    file, then holds a source-specific lock while opening and identity-checking the
    legacy source, copying from that bound descriptor, publishing, and atomically
    claiming the source under a private name for identity-safe cleanup. Immediately
    after acquiring that lock, queued callers reuse a matching published cache before
    inspecting any still-present legacy pathname. A replacement at the legacy path is
    restored without clobbering anything newer. Once publication commits, cleanup
    errors are warning-only and leave recoverable bytes at the public source, private
    candidate, or both. A missing/unreadable source or failed publication therefore
    leaves the legacy file untouched. If the legacy bytes don't match the current
    source's expected size (i.e. the legacy cache is stale or truncated), the legacy
    file is left untouched and a fresh cache is built from the source instead of
    reusing untrustworthy bytes.
    """
    source_fa = Path(source_fa)
    legacy_gvl = Path(legacy_gvl)
    gvlfa_dir = Path(gvlfa_dir)
    contigs = _contig_lengths(source_fa)  # raises here if source is unreadable
    fp = fingerprint(source_fa)
    source_lock = FileLock(f"{legacy_gvl}.lock")
    with source_lock:
        if meta := _matching_published_cache(gvlfa_dir, contigs, fp):
            return meta
        try:
            source_identity = legacy_gvl.stat()
        except FileNotFoundError:
            if meta := _matching_published_cache(gvlfa_dir, contigs, fp):
                return meta
            raise

        if source_identity.st_size != sum(contigs.values()):
            logger.info(
                f"Legacy cache {legacy_gvl} size does not match source {source_fa}; "
                "ignoring stale legacy bytes and building a fresh cache."
            )
            return build(source_fa, gvlfa_dir)

        meta_holder: dict[str, FastaCache] = {}
        with open(legacy_gvl, "rb") as legacy_source:
            opened_identity = os.fstat(legacy_source.fileno())
            if not _same_source_identity(source_identity, opened_identity):
                raise RuntimeError(
                    f"Legacy cache {legacy_gvl} changed before it could be opened "
                    "safely."
                )

            with atomic_dir(gvlfa_dir, overwrite=True) as tmp:
                with open(tmp / DATA_FILENAME, "wb") as staged_data:
                    shutil.copyfileobj(legacy_source, staged_data)
                copied_identity = os.fstat(legacy_source.fileno())
                if not _same_source_identity(source_identity, copied_identity):
                    raise RuntimeError(
                        f"Legacy cache {legacy_gvl} changed while it was being copied."
                    )
                meta = FastaCache(
                    format_version=FORMAT_VERSION,
                    genvarloader_version=_gvl_version(),
                    contigs=contigs,
                    source=_source_hints(source_fa, gvlfa_dir),
                    fingerprint=fp,
                )
                (tmp / METADATA_FILENAME).write_text(meta.model_dump_json())
                meta_holder["meta"] = meta

        _cleanup_legacy_source(legacy_gvl, source_identity)
        return meta_holder["meta"]


def _cache_dir_for(source_fa: Path) -> Path:
    return source_fa.with_name(source_fa.name + GVLFA_SUFFIX)


def _legacy_for(source_fa: Path) -> Path:
    return source_fa.with_name(source_fa.name + LEGACY_SUFFIX)


def is_gvlfa(path: str | Path) -> bool:
    """True if path is a .gvlfa cache directory."""
    path = Path(path)
    if not path.is_dir():
        return False
    return path.name.endswith(GVLFA_SUFFIX) or (path / METADATA_FILENAME).exists()


def ensure_cache(path: str | Path) -> tuple[FastaCache, Path]:
    """Resolve a usable cache for `path` (a .fa or a .gvlfa), building, migrating,
    or rebuilding as needed. Returns (metadata, path to sequence.bin)."""
    path = Path(path)
    if is_gvlfa(path):
        return _ensure_from_gvlfa(path)
    return _ensure_from_fasta(path)


def _ensure_from_fasta(source_fa: Path) -> tuple[FastaCache, Path]:
    gvlfa_dir = _cache_dir_for(source_fa)
    legacy = _legacy_for(source_fa)
    data_path = gvlfa_dir / DATA_FILENAME
    if gvlfa_dir.exists():
        try:
            meta: FastaCache | None = FastaCache.model_validate_json(
                (gvlfa_dir / METADATA_FILENAME).read_text()
            )
        except Exception:
            meta = None  # unreadable/corrupt metadata -> rebuild below
        if meta is not None:
            # Format-too-new raises here and propagates, matching _ensure_from_gvlfa:
            # gvl never silently downgrades a cache written by a newer version.
            _check_format_version(meta, gvlfa_dir)
            valid = _data_size_ok(gvlfa_dir, meta) and _fingerprints_match(
                meta.fingerprint, source_fa
            )
        else:
            valid = False
        if (
            not valid or meta is None
        ):  # redundant at runtime; satisfies the type checker
            logger.info(f"Building FASTA cache at {gvlfa_dir}.")
            meta = _ensure_built(source_fa, gvlfa_dir)
        return meta, data_path
    if legacy.exists():
        logger.info(f"Migrating legacy FASTA cache {legacy} -> {gvlfa_dir}.")
        return migrate_legacy(source_fa, legacy, gvlfa_dir), data_path
    logger.info(f"Building FASTA cache at {gvlfa_dir}.")
    return _ensure_built(source_fa, gvlfa_dir), data_path


def _ensure_from_gvlfa(gvlfa_dir: Path) -> tuple[FastaCache, Path]:
    data_path = gvlfa_dir / DATA_FILENAME
    try:
        meta, source, status = load(gvlfa_dir)
    except ValueError:
        raise  # format-too-new: actionable, do not swallow
    except Exception as e:
        raise ValueError(f"FASTA cache at {gvlfa_dir} is unreadable: {e}") from e
    if not _data_size_ok(gvlfa_dir, meta):
        if source is not None:
            return _ensure_built(source, gvlfa_dir), data_path
        raise ValueError(
            f"FASTA cache data at {data_path} is corrupt and the source FASTA "
            "could not be located to rebuild it."
        )
    if status == "stale":
        logger.info(f"Source FASTA changed; rebuilding cache at {gvlfa_dir}.")
        meta = _ensure_built(source, gvlfa_dir)
    elif status == "unvalidated":
        warnings.warn(
            f"Could not locate source FASTA for cache {gvlfa_dir}; using cached "
            "data without validation. On-demand reads (in_memory=False) will fail.",
            stacklevel=2,
        )
    return meta, data_path


def load(
    gvlfa_dir: str | Path,
) -> tuple[FastaCache, Path | None, Literal["fresh", "stale", "unvalidated"]]:
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
