"""On-disk layouts for the ``genotypes/svar2_ranges/`` var-key range cache.

A ``.svar2``-backed dataset caches, per ``(region, sample, ploid)``, the byte
window into its contig's var-key tables that the read-bound kernel must walk.
The obvious layout is a dense ``(R, S, P, 2)`` array -- and it is unusable at
cohort scale: 128 GB for one chromosome of All of Us, over 99% of it empty
(#357).

The **sparse** layout stores only the cells that hold a variant, in region-CSR
order. It is exact, not approximate, because an empty range carries no
information: ``gather_haps_readbound_impl`` slices ``positions[vs..ve]`` and
derives ``j = vs + k`` *inside* the loop body, so ``(0, 0)`` is byte-identical
to the true insertion point -- and strictly safer, since Rust slicing panics if
``vs > ve`` while ``(0, 0)`` is unconditionally in bounds. See the design spec
at ``docs/superpowers/specs/2026-09-14-svar2-sparse-range-cache-design.md``.

Both layouts live behind :class:`_RangeLookup`, resolved from a directory by
:func:`_ranges_reader`, so the reader, the writer and ``concat`` share one
definition of what these files mean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Iterator, Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ENTRY_DTYPE",
    "_RangeLookup",
    "_SparseRanges",
    "_DenseRanges",
    "_ranges_reader",
    "_SparseWriter",
    "nonempty_entries",
]

ENTRY_DTYPE = np.dtype(
    [
        ("snp_start", "<i8"),
        ("indel_start", "<i8"),
        ("snp_len", "<i4"),
        ("indel_len", "<i4"),
    ]
)
"""One sparse cache entry: 24 bytes, plus a 4-byte ``cell_id`` = 28 B per cell.

Dense costs 32 B per ``(region, sample, ploid)`` cell, so sparse is strictly
smaller at *every* fill level -- which is what lets this ship with no fill
threshold and no dense-fallback writer. Lengths rather than ends: one add on
gather, eight bytes saved per entry. Fields are ordered i8, i8, i4, i4 so the
record is naturally aligned and numpy adds no padding.
"""

ITER_BLOCK_ENTRIES = 1 << 20
"""Target entries per :meth:`_RangeLookup.iter_entries` block (~28 MB sparse).

A target, not a hard cap: a block never splits one region's cells across two
yields, so one region is the floor and a single region wider than this (only
possible if ``S * P > 2**20``) yields as one block larger than the target.
"""


class _RangeLookup(Protocol):
    """A var-key range cache, whatever its on-disk layout.

    Attributes:
        n_regions: ``R`` -- BED rows in the dataset.
        n_samples: ``S`` -- selected samples.
        ploidy: ``P``.
    """

    n_regions: int
    n_samples: int
    ploidy: int

    def lookup(
        self, r_q: NDArray[np.integer], si_q: NDArray[np.integer], P: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """The SNP and indel var-key ranges for a query block.

        Args:
            r_q: Region index per query. Must be in ``[0, n_regions)``.
            si_q: Sample slot per query, parallel to ``r_q``. In ``[0, n_samples)``.
            P: Ploidy; must equal :attr:`ploidy`.

        Returns:
            ``(vk_snp, vk_indel)``, each ``(len(r_q) * P, 2)`` C-contiguous int64
            in ``row = q * P + p`` order, which is what the Rust kernel expects.

        Raises:
            IndexError: If any region or sample index is out of bounds. A sparse
                miss is indistinguishable from an empty cell, so this cannot be
                left to fancy-indexing.
            ValueError: If ``r_q`` and ``si_q`` have different lengths, or ``P``
                does not equal :attr:`ploidy`.
        """
        ...

    def entries_for_regions(
        self, r0: int, r1: int
    ) -> tuple[NDArray[np.int64], NDArray[np.void]]:
        """Non-empty cells of regions ``[r0, r1)``, in ascending global-key order.

        The one primitive ``concat`` and :meth:`iter_entries` are both built on.
        Region-bounded rather than entry-bounded because ``concat`` merges by
        *merged region batch*: it needs "everything these regions hold", and a
        reader cannot answer that from an entry-count-bounded stream.

        Args:
            r0: First region, inclusive. Must be in ``[0, n_regions]``.
            r1: Last region, exclusive. Must be in ``[r0, n_regions]``.

        Returns:
            ``(key, entries)`` where ``key`` is int64
            ``r * (n_samples * ploidy) + slot * ploidy + ploid`` and ``entries``
            is a parallel :data:`ENTRY_DTYPE` array. Both are empty if the
            region range holds no non-empty cell.

        Raises:
            IndexError: If ``r0``/``r1`` violate ``0 <= r0 <= r1 <= n_regions``.
                An inverted or negative range must not be indistinguishable
                from a valid range that simply holds no cells.
        """
        ...

    def iter_entries(self) -> Iterator[tuple[NDArray[np.int64], NDArray[np.void]]]:
        """Non-empty cells in ascending global-key order, in blocks.

        A thin region-batched loop over :meth:`entries_for_regions`. Used by
        ``concat``'s dense-input path and by tests; never by the read path.

        Yields:
            ``(key, entries)``, as :meth:`entries_for_regions` returns them.
            Empty blocks are skipped.
        """
        ...


def _check_bounds(r_q: NDArray[np.integer], si_q: NDArray[np.integer], R: int, S: int):
    """Fail loudly on an out-of-range index.

    The dense layout got this free from fancy-indexing. Sparse does not: a probe
    that misses looks exactly like an absent (i.e. empty) cell, so an off-by-one
    would silently yield a reference-only haplotype instead of an IndexError.
    """
    if len(r_q) != len(si_q):
        raise ValueError(
            f"r_q and si_q must be parallel, got {len(r_q)} and {len(si_q)}"
        )
    if len(r_q) == 0:
        return
    if r_q.min() < 0 or r_q.max() >= R:
        raise IndexError(
            f"region index out of bounds: [{r_q.min()}, {r_q.max()}] not within [0, {R})"
        )
    if si_q.min() < 0 or si_q.max() >= S:
        raise IndexError(
            f"sample index out of bounds: [{si_q.min()}, {si_q.max()}] not within [0, {S})"
        )


@dataclass(slots=True)
class _SparseRanges:
    """Region-CSR table of non-empty ``(region, sample, ploid)`` cells.

    ``region_ptr[r]:region_ptr[r + 1]`` is region ``r``'s block; within a block,
    ``cell_id == slot * ploidy + ploid`` ascends, so a cell is found by bounded
    binary search. Search depth is fixed once from the table's WIDEST region
    block, not its average: one densely-filled region forces that many
    iterations for every query this table ever serves, including ones landing
    on sparse regions. ``region_ptr`` itself costs 1.6 MB genome-wide against a
    27 GB table.
    """

    region_ptr: NDArray[np.int64]
    cell_id: NDArray[np.int32]
    cell_vk: NDArray[np.void]
    n_regions: int
    n_samples: int
    ploidy: int
    _cell_span: int = field(init=False, repr=False, default=0)
    _depth: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        if len(self.region_ptr) != self.n_regions + 1:
            raise ValueError(
                f"region_ptr must have n_regions + 1 = {self.n_regions + 1} entries,"
                f" got {len(self.region_ptr)}"
            )
        if len(self.cell_id) != len(self.cell_vk):
            raise ValueError(
                f"cell_id ({len(self.cell_id)}) and cell_vk ({len(self.cell_vk)})"
                " must be parallel"
            )
        self._cell_span = self.n_samples * self.ploidy
        # Hoisted out of the probe. Computing this per lookup() is an O(R) pass
        # over a memmap-backed array: measured 1.5 us at R = 3,734 but 98 us at a
        # genome-scale R = 202,053, i.e. 2.9% of the whole 3.42 ms lookup budget
        # spent recomputing a constant. The same pass doubles as validation: a
        # truncated or mis-cumsum'd region_ptr is otherwise silent -- `lookup`
        # would return a wrong-but-plausible range instead of raising, because a
        # bad probe result looks exactly like a miss.
        w = self.region_ptr[1:] - self.region_ptr[:-1]
        if len(w) and w.min() < 0:
            bad = int(w.argmin())
            raise ValueError(
                "region_ptr must be non-decreasing, got region_ptr"
                f"[{bad}]={int(self.region_ptr[bad])} > region_ptr[{bad + 1}]="
                f"{int(self.region_ptr[bad + 1])}"
            )
        if len(self.region_ptr) and int(self.region_ptr[-1]) != len(self.cell_id):
            raise ValueError(
                f"region_ptr[-1] ({int(self.region_ptr[-1])}) must equal"
                f" len(cell_id) ({len(self.cell_id)}); the table is truncated"
            )
        widest = int(w.max()) if len(w) else 0
        # partition_point halves `size` to 1 in exactly ceil(log2(widest)) steps.
        self._depth = max(1, (widest - 1).bit_length())

    def lookup(
        self, r_q: NDArray[np.integer], si_q: NDArray[np.integer], P: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        if P != self.ploidy:
            raise ValueError(f"query ploidy {P} != cache ploidy {self.ploidy}")
        r_q = np.atleast_1d(np.asarray(r_q))
        si_q = np.atleast_1d(np.asarray(si_q))
        _check_bounds(r_q, si_q, self.n_regions, self.n_samples)

        n = len(r_q)
        vk_snp = np.zeros((n * P, 2), np.int64)
        vk_indel = np.zeros((n * P, 2), np.int64)
        if n == 0 or len(self.cell_id) == 0:
            return vk_snp, vk_indel

        cid = self.cell_id
        last = len(cid) - 1

        # int64 throughout: int32 si_q * P wraps silently and the later promotion
        # hides it, because the *wrapped* value is what gets promoted.
        # One broadcast rather than repeat + tile + add.
        target = (
            np.multiply(si_q, P, dtype=np.int64)[:, None] + np.arange(P, dtype=np.int64)
        ).reshape(-1)
        # region_ptr is gathered n times, not n * P times, then broadcast.
        r64 = np.asarray(r_q, np.int64)
        lo1 = self.region_ptr[r64]
        hi1 = self.region_ptr[r64 + 1]
        lo = np.repeat(lo1, P)

        if ((hi1 - lo1) == self._cell_span).all():
            # Every queried region block holds every cell, so cell_id is exactly
            # arange(S * P) there and position = lo + target with no search and no
            # hit test. This is the whole high-fill regime -- sequence-model
            # windows run at ~100% fill -- measured at 0.35 ms against 0.89 ms.
            pos = lo + target
            e = self.cell_vk[pos]
            vk_snp[:, 0] = e["snp_start"]
            np.add(e["snp_start"], e["snp_len"], out=vk_snp[:, 1])
            vk_indel[:, 0] = e["indel_start"]
            np.add(e["indel_start"], e["indel_len"], out=vk_indel[:, 1])
            return vk_snp, vk_indel

        hi = np.repeat(hi1, P)
        pos = self._lower_bound(lo, hi, target)
        # `lo <= pos` is what makes an empty trailing block (lo == hi == N, where
        # `base` had to be clamped) report a miss rather than matching the
        # previous region's last entry.
        hit = (lo <= pos) & (pos < hi)
        clamped = np.minimum(pos, last)
        np.logical_and(hit, cid[clamped] == target, out=hit)

        e = self.cell_vk[clamped]
        for out, start, length in (
            (vk_snp, e["snp_start"], e["snp_len"]),
            (vk_indel, e["indel_start"], e["indel_len"]),
        ):
            s = np.where(hit, start, 0)  # already int64; no .astype
            out[:, 0] = s
            np.add(s, np.where(hit, length, 0), out=out[:, 1])
        return vk_snp, vk_indel

    def _lower_bound(
        self, lo: NDArray[np.int64], hi: NDArray[np.int64], target: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """Per-element ``lower_bound`` of ``target`` in ``cell_id[lo:hi]``.

        ``np.searchsorted`` cannot express per-element bounds, so this is a
        manually vectorized search: the loop is over bit-depth, never over
        queries. It is the branchless ``std::partition_point`` form rather than
        the textbook ``(lo, hi)`` form for one reason -- it is *self-stabilizing*.
        Once a block's ``size`` reaches 1, ``half == 0``, ``mid == base`` and the
        update is a no-op, so extra iterations are free and no ``active = lo < hi``
        guard is needed. That drops the loop from twelve O(n) temporaries per
        iteration to two, and the whole probe by 1.13%.

        (The textbook form genuinely *needs* that guard: once ``lo == hi`` you get
        ``mid == lo``, and a true ``go`` sets ``lo = mid + 1 > hi``, so ``lo``
        drifts upward by one per remaining iteration. Do not delete the guard from
        that formulation -- this one removes the need for it instead.)

        The single ``np.minimum`` is needed only when the table's *last* region
        block is empty, which puts ``lo == len(cell_id)``; for every other block
        ``base`` stays within ``[lo, hi - 1]`` by construction. Padding ``cell_id``
        with a sentinel instead would force a full RAM copy of a memmap -- 27 GB
        genome-wide -- to avoid one clamp.
        """
        size = hi - lo
        base = np.minimum(lo, len(self.cell_id) - 1)
        cid = self.cell_id
        half = np.empty_like(size)
        mid = np.empty_like(size)
        go = np.empty(len(size), bool)
        for _ in range(self._depth):
            np.right_shift(size, 1, out=half)
            np.add(base, half, out=mid)
            np.less(cid[mid], target, out=go)
            base = np.where(go, mid, base)
            np.subtract(size, half, out=size)
        np.less(cid[base], target, out=go)
        base += go
        return base

    def entries_for_regions(
        self, r0: int, r1: int
    ) -> tuple[NDArray[np.int64], NDArray[np.void]]:
        """Non-empty cells of regions ``[r0, r1)``, key-ascending.

        The primitive both :meth:`iter_entries` and ``concat``'s region-batched
        merge are built on. Keys are dataset-global
        ``r * (n_samples * ploidy) + slot * ploidy + ploid``.

        Raises:
            IndexError: If ``r0``/``r1`` violate ``0 <= r0 <= r1 <= n_regions``.
        """
        if not 0 <= r0 <= r1 <= self.n_regions:
            raise IndexError(
                f"region range out of bounds: got r0={r0}, r1={r1}, expected"
                f" 0 <= r0 <= r1 <= n_regions={self.n_regions}"
            )
        a, b = int(self.region_ptr[r0]), int(self.region_ptr[r1])
        if b <= a:
            return np.empty(0, np.int64), np.empty(0, ENTRY_DTYPE)
        rows = np.repeat(
            np.arange(r0, r1, dtype=np.int64), np.diff(self.region_ptr[r0 : r1 + 1])
        )
        return (
            rows * self._cell_span + self.cell_id[a:b].astype(np.int64),
            np.asarray(self.cell_vk[a:b]),
        )

    def iter_entries(self) -> Iterator[tuple[NDArray[np.int64], NDArray[np.void]]]:
        rows = max(1, ITER_BLOCK_ENTRIES // max(self._cell_span, 1))
        for r0 in range(0, self.n_regions, rows):
            key, ent = self.entries_for_regions(r0, min(r0 + rows, self.n_regions))
            if len(key):
                yield key, ent


@dataclass(slots=True)
class _DenseRanges:
    """The legacy ``(R, S, P, 2)`` layout, kept so old datasets still open.

    Written by GVL <= 0.42.1 and by nothing since; ``gvl.write`` emits only the
    sparse layout. Re-running ``gvl.write`` is how an existing dataset is shrunk
    -- there is no migration tool (``gvl.migrate`` handles only the 1.x -> 2.0
    AoS-to-SoA track, ``_migrate.py:65``).
    """

    vk_snp_range: NDArray[np.int64]
    vk_indel_range: NDArray[np.int64]
    n_regions: int
    n_samples: int
    ploidy: int

    def __post_init__(self):
        expected = (self.n_regions, self.n_samples, self.ploidy, 2)
        for name, arr in (
            ("vk_snp_range", self.vk_snp_range),
            ("vk_indel_range", self.vk_indel_range),
        ):
            if arr.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected} = (n_regions, n_samples,"
                    f" ploidy, 2), got {arr.shape}"
                )

    def lookup(
        self, r_q: NDArray[np.integer], si_q: NDArray[np.integer], P: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        if P != self.ploidy:
            raise ValueError(f"query ploidy {P} != cache ploidy {self.ploidy}")
        r_q = np.atleast_1d(np.asarray(r_q))
        si_q = np.atleast_1d(np.asarray(si_q))
        # Fancy-indexing would raise on its own, but only for the region axis in
        # some shapes; check both so the two layouts fail identically. atleast_1d
        # matches _SparseRanges.lookup so a scalar query behaves the same on
        # either layout instead of raising only on dense.
        _check_bounds(r_q, si_q, self.n_regions, self.n_samples)
        snp = np.ascontiguousarray(
            np.asarray(self.vk_snp_range[r_q, si_q]).reshape(-1, 2), np.int64
        )
        indel = np.ascontiguousarray(
            np.asarray(self.vk_indel_range[r_q, si_q]).reshape(-1, 2), np.int64
        )
        return snp, indel

    def entries_for_regions(
        self, r0: int, r1: int
    ) -> tuple[NDArray[np.int64], NDArray[np.void]]:
        """Scan the dense arrays over ``[r0, r1)``, emitting non-empty cells.

        This has no analogue in the old code and is not free: a full pass reads
        the entire dense array once, which at All of Us chr22 is 128 GB. It
        exists solely so ``concat`` can merge a legacy dense shard into a sparse
        output, and it is region-bounded so ``concat`` can ask for exactly the
        merged batch it is assembling rather than driving a stream.

        Raises:
            IndexError: If ``r0``/``r1`` violate ``0 <= r0 <= r1 <= n_regions``.
        """
        if not 0 <= r0 <= r1 <= self.n_regions:
            raise IndexError(
                f"region range out of bounds: got r0={r0}, r1={r1}, expected"
                f" 0 <= r0 <= r1 <= n_regions={self.n_regions}"
            )
        span = self.n_samples * self.ploidy
        if r1 <= r0:
            return np.empty(0, np.int64), np.empty(0, ENTRY_DTYPE)
        snp = np.asarray(self.vk_snp_range[r0:r1])
        indel = np.asarray(self.vk_indel_range[r0:r1])
        ne = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
        # C-order nonzero => already ascending in (r, slot, ploid) = key order.
        ri, sj, pj = np.nonzero(ne)
        if len(ri) == 0:
            return np.empty(0, np.int64), np.empty(0, ENTRY_DTYPE)
        key = (r0 + ri).astype(np.int64) * span + sj.astype(np.int64) * self.ploidy + pj
        ent = np.empty(len(ri), ENTRY_DTYPE)
        ent["snp_start"] = snp[ri, sj, pj, 0]
        ent["snp_len"] = snp[ri, sj, pj, 1] - snp[ri, sj, pj, 0]
        ent["indel_start"] = indel[ri, sj, pj, 0]
        ent["indel_len"] = indel[ri, sj, pj, 1] - indel[ri, sj, pj, 0]
        return key, ent

    def iter_entries(self) -> Iterator[tuple[NDArray[np.int64], NDArray[np.void]]]:
        span = self.n_samples * self.ploidy
        rows = max(1, ITER_BLOCK_ENTRIES // max(span, 1))
        for r0 in range(0, self.n_regions, rows):
            key, ent = self.entries_for_regions(r0, min(r0 + rows, self.n_regions))
            if len(key):
                yield key, ent


def _raw(
    path: Path, dtype: "np.dtype[Any] | type[np.generic]", shape: tuple[int, ...]
) -> NDArray[Any]:
    """Memmap a raw headerless cache file, or an empty array if it holds nothing.

    ``np.memmap`` raises ``ValueError: cannot mmap an empty file`` on a 0-byte
    file, which a variant-free dataset or a per-contig shard can legitimately
    produce.

    Args:
        path: The raw ``tofile``-dumped file to open.
        dtype: The file's element dtype.
        shape: The file's shape, as recorded in ``svar2_meta.json``.

    Returns:
        A read-only memmap of ``path``, or an in-memory empty array of
        ``shape``/``dtype`` if ``shape`` has zero elements (no file needed).
    """
    n = int(np.prod(shape)) if len(shape) else 0
    if n == 0:
        return np.empty(shape, dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))


def _ranges_reader(ranges_dir: Path) -> _RangeLookup:
    """Open whichever range-cache layout is at ``ranges_dir``.

    A path-level factory rather than a method on ``Svar2Haps``: ``concat`` needs
    the same reader, and ``Svar2Haps.from_path`` additionally resolves and
    fingerprints the external ``.svar2`` store, which ``concat`` must not do.

    Args:
        ranges_dir: The dataset's ``genotypes/svar2_ranges/`` directory.

    Returns:
        A :class:`_SparseRanges` or :class:`_DenseRanges`.

    Raises:
        ValueError: If the meta's grid disagrees with the files beside it. A
            wrong ``n_samples`` makes every sparse probe miss, which reads as a
            silently variant-free dataset rather than an error.
    """
    ranges_dir = Path(ranges_dir)
    with open(ranges_dir / "svar2_meta.json") as f:
        meta = json.load(f)

    P = int(meta["ploidy"])
    R = int(meta["dense_snp_range"]["shape"][0])
    layout = meta.get("layout", "dense")

    if layout == "dense":
        # Pre-0.43.0 datasets carry S only in the vk array's shape.
        S = int(meta["vk_snp_range"]["shape"][1])
    elif layout == "sparse":
        S = int(meta["n_samples"])
        if int(meta["n_regions"]) != R:
            raise ValueError(
                f"svar2 cache meta is inconsistent: n_regions={meta['n_regions']} but"
                f" dense_snp_range has {R} rows."
            )
    else:
        raise ValueError(
            f"Unknown svar2 range cache layout {layout!r} at {ranges_dir}. This"
            " dataset was written by a newer GenVarLoader."
        )

    n_cols = len(np.load(ranges_dir / "sample_cols.npy"))
    if n_cols != S:
        raise ValueError(
            f"svar2 cache meta claims {S} samples but sample_cols.npy holds {n_cols}."
        )

    if layout == "dense":
        # vk_snp_range.npy / vk_indel_range.npy are raw headerless tofile dumps,
        # so meta["vk_snp_range"]["shape"] is the ONLY record of how to interpret
        # the bytes. np.memmap only raises if the file is too SHORT for the shape
        # we ask for, so a stale/wrong recorded shape that is too LONG would
        # otherwise be absorbed silently -- reading a truncated prefix of the
        # real grid with no error.
        for k in ("vk_snp_range", "vk_indel_range"):
            if tuple(meta[k]["shape"]) != (R, S, P, 2):
                raise ValueError(
                    f"svar2 cache meta: {k} shape {meta[k]['shape']} != grid"
                    f" {(R, S, P, 2)}"
                )
        return _DenseRanges(
            vk_snp_range=_raw(ranges_dir / "vk_snp_range.npy", np.int64, (R, S, P, 2)),
            vk_indel_range=_raw(
                ranges_dir / "vk_indel_range.npy", np.int64, (R, S, P, 2)
            ),
            n_regions=R,
            n_samples=S,
            ploidy=P,
        )

    n = int(meta["n_entries"])
    return _SparseRanges(
        # np.array, not np.asarray: np.asarray(memmap, np.int64) returns a VIEW
        # still backed by the mmap (np.shares_memory is True), so every lookup
        # would fancy-index through page faults and __post_init__ would scan the
        # file. region_ptr is 1.6 MB genome-wide against a 27 GB table -- read it
        # into RAM once.
        region_ptr=np.array(
            _raw(ranges_dir / "region_ptr.npy", np.int64, (R + 1,)), np.int64
        ),
        cell_id=_raw(ranges_dir / "cell_id.npy", np.int32, (n,)),
        cell_vk=_raw(ranges_dir / "cell_vk.npy", ENTRY_DTYPE, (n,)),
        n_regions=R,
        n_samples=S,
        ploidy=P,
    )


@dataclass(slots=True)
class _SparseWriter:
    """Streams a region-CSR table to ``region_ptr``/``cell_id``/``cell_vk``.

    Files are raw and headerless -- the existing convention for everything in
    ``svar2_ranges/`` except ``sample_cols.npy`` -- so they can simply be
    appended to. A real ``.npy`` would need a placeholder header pre-written and
    patched at the end, because ``N`` is unknown until the last contig is done.

    The same consequence applies to ``svar2_meta.json``: it can only be written
    *after* the loop, so an aborted write leaves data files with no meta. That is
    safe only because ``write`` builds into an ``atomic_dir`` tmp that is
    discarded on failure.

    Use it as a context manager. ``region_ptr`` is published only on clean exit,
    because a ``region_ptr`` written from a ``finally`` would index a truncated
    ``cell_id``/``cell_vk`` -- a dataset that opens and silently returns garbage
    rather than one that fails.
    """

    ranges_dir: "Path"
    n_samples: int
    ploidy: int
    n_entries: int = 0
    # Every attribute must be declared: `slots=True` gives the class no __dict__,
    # so an undeclared `self._span = ...` in __post_init__ raises AttributeError.
    _span: int = field(init=False, repr=False, default=0)
    _ptr: "list[NDArray[np.int64]]" = field(
        init=False, repr=False, default_factory=list
    )
    _regions_done: int = field(init=False, repr=False, default=0)
    _f_cell: "IO[bytes]" = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    _f_vk: "IO[bytes]" = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self):
        span = self.n_samples * self.ploidy
        if span >= 2**31:
            raise ValueError(
                f"n_samples * ploidy = {span} does not fit int32, so the sparse"
                " range cache cannot address a cell. Shard the samples."
            )
        self._span = span
        self._ptr = [np.zeros(1, np.int64)]
        self._f_cell = open(self.ranges_dir / "cell_id.npy", "wb")
        self._f_vk = open(self.ranges_dir / "cell_vk.npy", "wb")

    def __enter__(self) -> "_SparseWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Close the handles unconditionally: `atomic_dir` discards the tmp tree on
        # failure, but the handles are pinned by the propagating traceback's frame
        # until GC, which on a long write is an open-fd leak per shard.
        if exc_type is None:
            self.close()
        else:
            self._close_files()
        return False

    def _counts(self, r: NDArray[np.integer], rc: int) -> NDArray[np.int64]:
        """Per-region entry counts, with the overshoot guard.

        ``np.bincount(x, minlength=rc)`` silently returns a **longer** array when
        an index exceeds ``rc`` (index 9 with ``minlength=5`` gives 10 bins).
        Unchecked, that lengthens ``region_ptr`` past ``R + 1`` while the meta
        still declares ``[R + 1]``, and the reader memmaps a truncated prefix --
        a silently corrupt dataset rather than a crash. Keys must be contig-local.
        """
        cnt = np.bincount(r, minlength=rc)
        if len(cnt) != rc:
            raise ValueError(
                f"svar2 range cache got region index {len(cnt) - 1} for a contig"
                f" of {rc} regions; indices must be contig-local."
            )
        return cnt.astype(np.int64, copy=False)

    def append(self, key: NDArray[np.int64], ent: NDArray[np.void], lo: int, rc: int):
        """Append one already-ordered block of entries.

        The key-based entry point, used by ``concat``'s merge. The write path
        calls :meth:`append_contig` instead, which does the ordering itself.

        Args:
            key: Strictly ascending **region-local** keys,
                ``(r - lo) * n_samples * ploidy + slot * ploidy + ploid``.
            ent: Parallel :data:`ENTRY_DTYPE` entries.
            lo: The contig's first region index in the dataset.
            rc: The contig's region count.

        Raises:
            ValueError: If the caller's regions are not contiguous and in order,
                if the keys are not strictly ascending, or if a key is out of
                range for this block. Contiguity is the one load-bearing
                invariant of the write path: blocks from ``bed.partition_by``
                must partition ``[0, R)`` in the same order as the running
                ``contig_offset``. Asserting it directly covers every way a
                future bed could break it.
        """
        self._check_lo(lo)
        if len(key) and not np.all(np.diff(key) > 0):
            raise ValueError("svar2 range cache entries are not strictly ascending")

        ri = (key // self._span).astype(np.int64)
        cnt = self._counts(ri, rc)
        np.asarray(key % self._span, np.int32).tofile(self._f_cell)
        np.asarray(ent, ENTRY_DTYPE).tofile(self._f_vk)

        self._ptr.append(self.n_entries + cnt.cumsum())
        self.n_entries += len(key)
        self._regions_done += rc

    def append_contig(
        self,
        regions: "list[NDArray[np.int32]]",
        cells: "list[NDArray[np.int32]]",
        ents: "list[NDArray[np.void]]",
        lo: int,
        rc: int,
    ) -> None:
        """Merge one contig's per-chunk blocks into region-major order and append.

        Each block from :func:`nonempty_entries` is already region-major, and
        chunk ``i``'s sample slots lie entirely below chunk ``i + 1``'s, so the
        merged order is fixed by region alone. That makes this a stable counting
        sort with ``O(rc)`` of auxiliary state, not a comparison sort.

        This is not a micro-optimization over ``np.argsort(key, kind="stable")``:
        numpy maps ``kind="stable"`` to radix **only** for integer types of 16
        bits or fewer, whatever its docstring says. int32 and int64 get timsort,
        ``O(N log N)`` -- measured 98 and 136 ns/element on random input against
        2.9 ns for int16. An int64 key sort only *looks* linear here because
        timsort's run detection fires on the per-chunk runs, and that degrades as
        chunks get smaller, which is exactly the regime the sort was chosen to
        survive. Measured at ``N = 18e6``: 914 ms (k=30) / 1369 ms (k=500) for the
        sort against 532 / 528 ms here, byte-identical output.

        Args:
            regions: Per-chunk contig-local region indices, chunks in ascending
                sample order.
            cells: Parallel ``slot * ploidy + ploid`` values.
            ents: Parallel :data:`ENTRY_DTYPE` entries.
            lo: The contig's first region index in the dataset.
            rc: The contig's region count.

        Raises:
            ValueError: If the caller's regions are not contiguous and in order,
                or if a region index is out of range for this contig.
        """
        self._check_lo(lo)

        # Pass 1: per-region totals. O(rc) of state -- never (n_chunks x rc),
        # which is what makes this safe at samples_per_chunk == 1 (535k chunks at
        # cohort scale).
        total = np.zeros(rc, np.int64)
        for r in regions:
            total += self._counts(r, rc)

        n = int(total.sum())
        cursor = np.empty(rc, np.int64)
        cursor[0] = 0
        np.cumsum(total[:-1], out=cursor[1:])

        # Pass 2: scatter each chunk to its final offsets. `arange - start[r]` is
        # the within-region rank, valid because each block is region-grouped.
        out_cell = np.empty(n, np.int32)
        out_ent = np.empty(n, ENTRY_DTYPE)
        for r, c, e in zip(regions, cells, ents):
            cnt = self._counts(r, rc)
            start = np.empty(rc, np.int64)
            start[0] = 0
            np.cumsum(cnt[:-1], out=start[1:])
            dst = cursor[r]
            dst += np.arange(len(r), dtype=np.int64)
            dst -= start[r]
            out_cell[dst] = c
            out_ent[dst] = e
            cursor += cnt

        out_cell.tofile(self._f_cell)
        out_ent.tofile(self._f_vk)
        self._ptr.append(self.n_entries + total.cumsum())
        self.n_entries += n
        self._regions_done += rc

    def _check_lo(self, lo: int) -> None:
        if lo != self._regions_done:
            raise ValueError(
                f"svar2 range cache requires contiguous region blocks in order:"
                f" got a block starting at region {lo} after {self._regions_done}"
                f" regions. Is the bed still contig-grouped (sp.bed.sort)?"
            )

    def _close_files(self) -> None:
        try:
            self._f_cell.close()
        finally:
            self._f_vk.close()

    def close(self) -> int:
        """Close the data files, publish ``region_ptr``, and return ``N``."""
        self._close_files()
        np.concatenate(self._ptr).astype(np.int64).tofile(
            self.ranges_dir / "region_ptr.npy"
        )
        return self.n_entries


def nonempty_entries(
    snp: NDArray[np.int64], indel: NDArray[np.int64], slot0: int, ploidy: int
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.void]]:
    """Extract non-empty cells from a ``(rc, ns, P, 2)`` pair of range blocks.

    Args:
        snp: SNP ranges, ``(rc, ns, P, 2)`` -- normally a ``transpose(2, 0, 1, 3)``
            view of a hap-major genoray chunk.
        indel: Indel ranges, same shape.
        slot0: Dataset sample slot of this block's first column.
        ploidy: ``P``.

    Returns:
        ``(region, cell, entries)``, region-major: ``region`` is **contig-local**
        and non-decreasing, ``cell`` is ``slot * ploidy + ploid`` and ascends
        within each region. Split rather than combined into one key because
        :meth:`_SparseWriter.append_contig` needs the region axis on its own to
        count, and ``cell`` is what lands on disk -- combining them would only be
        undone again.
    """
    ne = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
    # np.nonzero walks the LOGICAL shape in C order, so (r, slot, ploid) comes
    # out ascending even though `ne` is NOT C-contiguous: the `>` above inherits
    # the transposed view's stride permutation, because numpy allocates ufunc
    # output with NPY_KEEPORDER. Do not "fix" that with ascontiguousarray --
    # materializing (rc, ns, P) in C order is a strided scatter costing ~11x the
    # comparison itself (97.7 ms vs 8.8 ms on a 15e6-cell chunk).
    ri, sj, pj = np.nonzero(ne)
    ent = np.empty(len(ri), ENTRY_DTYPE)
    ent["snp_start"] = snp[ri, sj, pj, 0]
    ent["snp_len"] = snp[ri, sj, pj, 1] - snp[ri, sj, pj, 0]
    ent["indel_start"] = indel[ri, sj, pj, 0]
    ent["indel_len"] = indel[ri, sj, pj, 1] - indel[ri, sj, pj, 0]
    return ri.astype(np.int32), ((slot0 + sj) * ploidy + pj).astype(np.int32), ent
