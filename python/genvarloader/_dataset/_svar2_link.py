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


#: Per-contig subtrees whose bytes a dataset actually reads. Anything else in
#: a contig directory belongs to some other pipeline and is none of this
#: fingerprint's business -- see `_svar2_payload_files`.
_PAYLOAD_SUBTREES: tuple[str, ...] = ("dense", "fields", "indel", "var_key")

#: Per-contig payload files that sit directly in the contig directory.
_PAYLOAD_FILES: tuple[str, ...] = ("max_del.npy",)

_PAYLOAD_SUFFIXES = frozenset({".bin", ".npy"})


def _svar2_payload_files(svar2_path: Path) -> list[Path]:
    """The ``.bin``/``.npy`` files under the payload subtrees, sorted.

    Deliberately NOT a walk of the whole store. A ``.svar2`` store is written
    by one pipeline and read by many, so it accumulates sibling layers --
    per-contig annotations, scratch indices -- that no dataset opens. Counting
    those makes an unrelated pipeline's write break every reader of the store
    (issue #419), and it does so in the direction that matters least: the
    variant data is byte-for-byte unchanged while every consumer goes down.

    Scoping to the read payload keeps the documented contract exactly -- the
    fingerprint still changes iff the data the dataset reads changes -- while
    additive siblings become invisible, which is the correct behavior.
    """
    files: list[Path] = []
    for contig in sorted(p for p in svar2_path.iterdir() if p.is_dir()):
        for name in _PAYLOAD_FILES:
            f = contig / name
            if f.is_file() and f.suffix in _PAYLOAD_SUFFIXES:
                files.append(f)
        for sub in _PAYLOAD_SUBTREES:
            d = contig / sub
            if not d.is_dir():
                continue
            files.extend(
                p for p in d.rglob("*") if p.is_file() and p.suffix in _PAYLOAD_SUFFIXES
            )
    return sorted(files)


def _svar2_ignored_files(svar2_path: Path) -> list[Path]:
    """``.bin``/``.npy`` files in the store that the payload scope excludes.

    Used only to explain a mismatch. A large ignored set is the signature of
    a sibling annotation layer, and saying so turns issue #419's manual scope
    bisection into a line of the error message.
    """
    payload = set(_svar2_payload_files(svar2_path))
    return sorted(
        p
        for p in svar2_path.rglob("*")
        if p.is_file() and p.suffix in _PAYLOAD_SUFFIXES and p not in payload
    )


def _svar2_legacy_store_fingerprint(svar2_path: Path) -> tuple[int, int]:
    """The pre-scoping fingerprint: every ``.bin``/``.npy`` in the store.

    Kept solely to honor links recorded before the scope was narrowed. Those
    records counted sibling layers, so re-scoping would invalidate every one
    of them -- a check that starts rejecting stores it accepted yesterday is
    worse than the bug being fixed.
    """
    files = [
        p
        for p in svar2_path.rglob("*")
        if p.is_file() and p.suffix in _PAYLOAD_SUFFIXES
    ]
    return len(files), sum(p.stat().st_size for p in files)


def _svar2_store_fingerprint(svar2_path: Path) -> tuple[int, int]:
    """Deterministic (file count, total bytes) over the .svar2 store's data files.

    Covers the payload subtrees only (dense + fields + var_key + long-allele
    across all contigs, plus each contig's ``max_del.npy``). Changes iff the
    store's data files change -- that is this fingerprint's only contract, and
    scoping the walk is what makes the implementation match it.
    """
    files = _svar2_payload_files(svar2_path)
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

    exp = link.fingerprint

    n_files_observed, bytes_observed = _svar2_store_fingerprint(svar2_path)
    if (n_files_observed, bytes_observed) == (exp.n_files, exp.store_bytes):
        return

    # A link recorded before the fingerprint was scoped counted every
    # .bin/.npy in the store, sibling layers included. Accept that shape too:
    # narrowing the scope fixes issue #419 for stores that GAIN a layer, and
    # it must not break the datasets that verify cleanly today.
    if _svar2_legacy_store_fingerprint(svar2_path) == (exp.n_files, exp.store_bytes):
        return

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
            f"svar2 fingerprint mismatch at {svar2_path}: "
            + "; ".join(mismatches)
            + _mismatch_diagnosis(svar2_path, exp)
        )


def _mismatch_diagnosis(svar2_path: Path, exp: Svar2Fingerprint) -> str:
    """Context appended to a mismatch, so the message is actionable on its own.

    The person who has to diagnose this is whoever's job broke, not whoever
    changed the store, and they get one traceback to work from. Byte counts
    alone forced a manual scope bisection (issue #419), so state the scope
    that was used, what it covered, and -- the part that actually identifies
    a sibling-layer situation -- what it ignored.
    """
    try:
        payload = _svar2_payload_files(svar2_path)
        ignored = _svar2_ignored_files(svar2_path)
    except OSError as exc:  # unreadable store: say so rather than masking it
        return f"\n  (could not scan the store to explain this: {exc})"

    lines = [
        "",
        f"  scope: payload only -- each contig's {list(_PAYLOAD_FILES)} plus "
        f"{list(_PAYLOAD_SUBTREES)}",
        f"  payload files counted: {len(payload)} "
        f"({sum(p.stat().st_size for p in payload)} bytes)",
    ]
    if ignored:
        example = ignored[0].relative_to(svar2_path)
        lines.append(
            f"  non-payload files ignored: {len(ignored)} "
            f"({sum(p.stat().st_size for p in ignored)} bytes), e.g. {example}"
        )
        lines.append(
            "  a sibling layer like that does NOT affect this fingerprint; if "
            "the recorded value was written before the fingerprint was scoped "
            "to the payload, it counted those files and must be re-recorded."
        )
    elif exp.n_files > len(payload) and exp.store_bytes > sum(
        q.stat().st_size for q in payload
    ):
        # A record written under the old unscoped walk counted a SUPERSET of
        # the payload, so it is larger on both axes. A store that has since
        # lost the sibling layer has nothing left to point at, which is how
        # this looks from the consumer's side -- and calling that corruption
        # would send someone hunting for damage that is not there.
        lines.append(
            "  the recorded value is larger than the payload on both counts "
            "and no sibling layer remains to account for it. That is what a "
            "record written before the fingerprint was scoped looks like once "
            "the layer is gone; it is also what real truncation looks like. "
            "Compare against a replica that still carries the layer before "
            "deciding, then either restore the store or re-record the link."
        )
    else:
        lines.append(
            "  no non-payload files present, so this mismatch is in the data "
            "the dataset actually reads -- treat the store as modified."
        )
    return "\n".join(lines)


def make_svar2_link(gvl_path: Path, svar2_path: Path) -> Svar2Link:
    """Build a :class:`Svar2Link` recording the on-disk relationship between a gvl dataset and the ``.svar2`` store it reads from.

    Args:
        gvl_path: Path to the gvl dataset directory.
        svar2_path: Path to the ``.svar2`` store the dataset links to.

    Returns:
        Svar2Link: Relative/absolute paths to the store plus its fingerprint
            (file count and total byte size), used to detect a moved or
            modified store on open.
    """
    svar2_resolved = svar2_path.resolve()
    n_files, store_bytes = _svar2_store_fingerprint(svar2_resolved)
    return Svar2Link(
        relative_path=os.path.relpath(svar2_resolved, start=gvl_path).replace(
            os.sep, "/"
        ),
        absolute_path=str(svar2_resolved),
        fingerprint=Svar2Fingerprint(n_files=n_files, store_bytes=store_bytes),
    )
