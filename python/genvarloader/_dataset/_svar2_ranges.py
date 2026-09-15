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

from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["ENTRY_DTYPE", "_RangeLookup", "_SparseRanges"]

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
