"""Resolution and integrity for the GVL dataset -> .svar2 back-reference.

Mirrors _svar_link.py; the fingerprint keys on the .svar2 store's stable
identity (file count + summed byte size of its data files) rather than
SVAR1's variant_idxs.npy / index.arrow, neither of which .svar2 has.
SparseVar2 exposes no cheap variant-count accessor, so a semantic
n_variants field is deliberately not part of this fingerprint -- deriving
one would require contig lengths plus a full-span decode, which is
over-engineering for an integrity check.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class Svar2Fingerprint(BaseModel):
    n_files: int
    store_bytes: int


class Svar2Link(BaseModel):
    relative_path: str
    absolute_path: str
    fingerprint: Svar2Fingerprint


def _svar2_store_fingerprint(svar2_path: Path) -> tuple[int, int]:
    """Deterministic (file count, total bytes) over the .svar2 store's data files.

    Walks the store for ``.bin``/``.npy`` files (dense + var_key + long-allele
    payloads across all contigs). Changes iff the store's data files change --
    that is this fingerprint's only contract.
    """
    files = sorted(
        p for p in svar2_path.rglob("*") if p.is_file() and p.suffix in {".bin", ".npy"}
    )
    return len(files), sum(p.stat().st_size for p in files)


def _resolve_svar2(
    gvl_path: Path,
    link: Svar2Link | None,
    override: Path | str | None,
) -> Path:
    """Resolve the .svar2 directory referenced by a GVL dataset.

    Order: override -> link.relative_path -> link.absolute_path -> sibling *.svar2.
    Raises FileNotFoundError if none resolve to a directory.
    """
    if override is not None:
        p = Path(override)
        if not p.is_dir():
            raise FileNotFoundError(
                f"svar2 override path does not exist or is not a directory: {p}"
            )
        return p

    if link is not None:
        rel = (gvl_path / link.relative_path).resolve()
        if rel.is_dir():
            return rel
        absp = Path(link.absolute_path)
        if absp.is_dir():
            return absp

    siblings = sorted(gvl_path.parent.glob("*.svar2"))
    if len(siblings) == 1:
        return siblings[0]

    expected = Path(link.absolute_path).name if link is not None else "<unknown>.svar2"
    raise FileNotFoundError(
        f"Could not locate svar2 '{expected}' for GVL dataset at {gvl_path}. "
        f"Tried: stored relative path, stored absolute path, sibling *.svar2. "
        f"Pass `svar2=` to `Dataset.open(...)` to override."
    )


def _verify_svar2_fingerprint(svar2_path: Path, link: Svar2Link | None) -> None:
    """Compare the recorded fingerprint against the resolved svar2 store.

    No-op when ``link`` is None (legacy dataset, or one without a svar2 link).
    Raises ValueError on mismatch.
    """
    if link is None:
        return

    n_files_observed, bytes_observed = _svar2_store_fingerprint(svar2_path)

    exp = link.fingerprint
    mismatches: list[str] = []
    if n_files_observed != exp.n_files:
        mismatches.append(
            f"n_files: expected {exp.n_files}, observed {n_files_observed}"
        )
    if bytes_observed != exp.store_bytes:
        mismatches.append(
            f"store_bytes: expected {exp.store_bytes}, observed {bytes_observed}"
        )
    if mismatches:
        raise ValueError(
            f"svar2 fingerprint mismatch at {svar2_path}: " + "; ".join(mismatches)
        )


def make_svar2_link(gvl_path: Path, svar2_path: Path) -> Svar2Link:
    svar2_resolved = svar2_path.resolve()
    n_files, store_bytes = _svar2_store_fingerprint(svar2_resolved)
    return Svar2Link(
        relative_path=os.path.relpath(svar2_resolved, start=gvl_path).replace(
            os.sep, "/"
        ),
        absolute_path=str(svar2_resolved),
        fingerprint=Svar2Fingerprint(n_files=n_files, store_bytes=store_bytes),
    )
