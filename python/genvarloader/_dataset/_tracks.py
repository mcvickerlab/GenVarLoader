from __future__ import annotations

import enum
import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import awkward as ak
import numba as nb
import numpy as np
from einops import repeat
from numpy.typing import NDArray
from seqpro.rag import Ragged

from .._flat import _Flat
from .._ragged import INTERVAL_DTYPE, FlatIntervals, RaggedIntervals, RaggedTracks
from .._utils import lengths_to_offsets
from ._indexing import DatasetIndexer
from ._insertion_fill import InsertionFill, Repeat5p
from ._intervals import intervals_to_tracks
from ._protocol import Reconstructor
from ._splice import SplicePlan

if TYPE_CHECKING:
    from ._haps import Haps

# Strategy enum (mirrors _insertion_fill.py; duplicated to avoid Python-level
# imports inside @njit functions)
_REPEAT_5P = 0
_REPEAT_5P_NORM = 1
_CONSTANT = 2
_FLANK_SAMPLE = 3
_INTERPOLATE = 4


@nb.njit(nogil=True, cache=True, inline="always")
def _xorshift64(x: np.uint64) -> np.uint64:
    """Single round of xorshift64. Pure function — safe in parallel."""
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    return x


@nb.njit(nogil=True, cache=True, inline="always")
def _hash4(a: np.uint64, b: np.uint64, c: np.uint64, d: np.uint64) -> np.uint64:
    """Hash four uint64 values into one. Used as a per-position deterministic seed."""
    h = a
    h = _xorshift64(h ^ b)
    h = _xorshift64(h ^ c)
    h = _xorshift64(h ^ d)
    return h


@nb.njit(nogil=True, cache=True, inline="always")
def _apply_insertion_fill(
    out: NDArray[np.floating],
    out_idx: int,
    writable_length: int,
    v_len: int,
    track: NDArray[np.floating],
    v_rel_pos: int,
    strategy_id: int,
    params: NDArray[np.float64],
    base_seed: np.uint64,
    query: int,
    hap: int,
):
    """Write `writable_length` values at out[out_idx:] according to strategy.

    v_len is the total length of the insertion stretch (v_diff + 1); the kernel
    may truncate the actual write to writable_length when running out of output.
    """
    track_len = len(track)

    # The _REPEAT_5P branch is unreachable from the outer kernel (which short-circuits
    # this strategy before calling). Kept for completeness and direct-helper-call safety.
    if strategy_id == _REPEAT_5P:
        val = track[v_rel_pos]
        for i in range(writable_length):
            out[out_idx + i] = val

    elif strategy_id == _REPEAT_5P_NORM:
        val = track[v_rel_pos] / v_len
        for i in range(writable_length):
            out[out_idx + i] = val

    elif strategy_id == _CONSTANT:
        val = params[0]
        for i in range(writable_length):
            out[out_idx + i] = val

    elif strategy_id == _FLANK_SAMPLE:
        width = np.int64(params[0])
        pool_lo = max(0, v_rel_pos - width)
        pool_hi = min(track_len - 1, v_rel_pos + width)
        pool_size = pool_hi - pool_lo + 1
        for i in range(writable_length):
            seed = _hash4(
                base_seed,
                np.uint64(query),
                np.uint64(hap),
                np.uint64(out_idx + i),
            )
            offset = np.int64(seed % np.uint64(pool_size))
            out[out_idx + i] = track[pool_lo + offset]

    elif strategy_id == _INTERPOLATE:
        order = np.int64(params[0])
        # Number of anchor values per side: ceil((order+1)/2)
        k = (order + 1 + 1) // 2  # ceil((order+1)/2)
        # Anchors: 5' side at x = 0, -1, -2, ...; 3' side at x = v_len, v_len+1, ...
        n_anchors = 2 * k
        xs = np.empty(n_anchors, dtype=np.float64)
        ys = np.empty(n_anchors, dtype=np.float64)
        for j in range(k):
            ref_idx = v_rel_pos - j
            ref_idx = max(ref_idx, 0)
            xs[j] = -float(j)
            ys[j] = track[ref_idx]
        for j in range(k):
            ref_idx = v_rel_pos + 1 + j
            ref_idx = min(ref_idx, track_len - 1)
            xs[k + j] = float(v_len) + float(j)
            ys[k + j] = track[ref_idx]
        # Lagrange interpolation at each output position in [0, writable_length)
        for i in range(writable_length):
            x = float(i)
            acc = 0.0
            for a in range(n_anchors):
                term = ys[a]
                for b in range(n_anchors):
                    if b == a:
                        continue
                    term *= (x - xs[b]) / (xs[a] - xs[b])
                acc += term
            out[out_idx + i] = acc


@nb.njit(parallel=True, nogil=True, cache=True)
def shift_and_realign_tracks_sparse(
    out: NDArray[np.floating],
    out_offsets: NDArray[np.integer],
    regions: NDArray[np.integer],
    shifts: NDArray[np.integer],
    geno_offset_idx: NDArray[np.integer],
    geno_v_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    v_starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    tracks: NDArray[np.floating],
    track_offsets: NDArray[np.integer],
    params: NDArray[np.float64],
    keep: NDArray[np.bool_] | None = None,
    keep_offsets: NDArray[np.integer] | None = None,
    strategy_id: int = 0,
    base_seed: np.uint64 = np.uint64(0),
):
    """Shift and realign tracks to correspond to haplotypes.

    Parameters
    ----------
    out : NDArray[np.float32]
        Ragged array with shape (batch, ploidy). Shifted and re-aligned tracks.
    out_offsets : NDArray[np.int64]
        Shape = (batch*ploidy + 1) Offsets into out.
    regions : NDArray[np.int32]
        Shape = (batch, 3) Regions, each is (contig_idx, start, end).
    shifts : NDArray[np.int32]
        Shape = (batch, ploidy) Shifts for each haplotype.
    geno_offset_idx : NDArray[np.intp]
        Shape = (batch, ploidy) Indices into offsets for each region.
    geno_v_idxs : NDArray[np.int32]
        Shape = (variants) Indices of variants.
    geno_offsets : NDArray[np.uint32]
        Shape = (tot_regions*samples*ploidy + 1) Offsets into variant idxs.
    positions : NDArray[np.int32]
        Shape = (total_variants) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (total_variants) Sizes of variants.
    tracks : NDArray[np.float32]
        Shape = (batch*ploidy*length) Tracks.
    track_offsets : NDArray[np.int64]
        Shape = (batch + 1) Offsets into tracks.
    keep : Optional[NDArray[np.bool_]]
        Shape = (batch*ploidy*variants) Keep mask for genotypes.
    keep_offsets : Optional[NDArray[np.int64]]
        Shape = (batch*ploidy + 1) Offsets into keep.
    """
    n_regions, ploidy = geno_offset_idx.shape
    for query in nb.prange(n_regions):
        t_s, t_e = track_offsets[query], track_offsets[query + 1]
        q_track = tracks[t_s:t_e]
        # assumes start is never altered upstream by differing hap lengths (true for left-aligned variants)
        q_start = regions[query, 1]

        for hap in nb.prange(ploidy):
            o_idx = geno_offset_idx[query, hap]

            k_idx = query * ploidy + hap
            if keep is not None and keep_offsets is not None:
                qh_keep = keep[keep_offsets[k_idx] : keep_offsets[k_idx + 1]]
            else:
                qh_keep = None

            out_s, out_e = out_offsets[k_idx], out_offsets[k_idx + 1]
            qh_out = out[out_s:out_e]
            qh_shifts = shifts[query, hap]

            shift_and_realign_track_sparse(
                offset_idx=o_idx,
                geno_v_idxs=geno_v_idxs,
                geno_offsets=geno_offsets,
                v_starts=v_starts,
                ilens=ilens,
                shift=qh_shifts,
                track=q_track,
                query_start=q_start,
                out=qh_out,
                params=params,
                keep=qh_keep,
                strategy_id=strategy_id,
                base_seed=base_seed,
                query=query,
                hap=hap,
            )


@nb.njit(nogil=True, cache=True)
def shift_and_realign_track_sparse(
    offset_idx: int,
    geno_v_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    v_starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    shift: int,
    track: NDArray[np.floating],
    query_start: int,
    out: NDArray[np.floating],
    params: NDArray[np.float64],
    keep: NDArray[np.bool_] | None = None,
    strategy_id: int = 0,
    base_seed: np.uint64 = np.uint64(0),
    query: int = 0,
    hap: int = 0,
):
    """Shift and realign a track to correspond to a haplotype.

    Parameters
    ----------
    offset_idx : NDArray[np.int32]
        Shape = (n_variants) Genotypes of variants.
    positions : NDArray[np.int32]
        Shape = (total_variants) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (total_variants) Sizes of variants.
    shift : int
        Total amount to shift by.
    track : NDArray[np.float32]
        Shape = (length) Track.
    out : NDArray[np.uint8]
        Shape = (out_length) Shifted and re-aligned track.
    keep : Optional[NDArray[np.bool_]]
        Shape = (n_variants) Keep mask for genotypes.
    """
    if geno_offsets.ndim == 1:
        o_s, o_e = geno_offsets[offset_idx], geno_offsets[offset_idx + 1]
    else:
        o_s, o_e = geno_offsets[:, offset_idx]
    _variant_idxs = geno_v_idxs[o_s:o_e]
    length = len(out)
    n_variants = len(_variant_idxs)

    if n_variants == 0:
        # guaranteed to have shift = 0
        out[:] = track[:length]
        return

    # where to get next track value
    track_idx = 0
    # where to put next value
    out_idx = 0
    # how much we've shifted
    shifted = 0

    for v in range(n_variants):
        if keep is not None and not keep[v]:
            continue

        variant: np.int32 = _variant_idxs[v]

        # position of variant relative to ref from fetch(contig, start, q_end)
        # i.e. has been put into same coordinate system as ref_idx
        v_rel_pos = v_starts[variant] - query_start
        v_diff = ilens[variant]
        # +1 assumes atomized variants, exactly 1 nt shared between REF and ALT
        v_rel_end = v_rel_pos - min(0, v_diff) + 1

        # variant is a DEL spanning start
        if v_diff < 0 and v_rel_pos < 0 and v_rel_end >= 0:
            track_idx = v_rel_end
            continue

        # overlapping variants
        # v_rel_pos < ref_idx only if we see an ALT at a given position a second
        # time or more. We'll do what bcftools consensus does and only use the
        # first ALT variant we find.
        if v_rel_pos < track_idx:
            continue

        v_len = max(0, v_diff) + 1

        # handle shift
        if shifted < shift:
            ref_shift_dist = v_rel_pos - track_idx
            # need more than variant to finish shift
            if shifted + ref_shift_dist + v_len < shift:
                # skip the variant
                continue
            # can finish shift without using variant
            elif shifted + ref_shift_dist >= shift:
                track_idx += shift - shifted
                shifted = shift
                # can still use the variant and whatever ref is left between
                # ref_idx and the variant
            # ref + (some of) variant is enough to finish shift
            else:
                # how much left to shift - amount of ref we can use
                allele_start_idx = shift - shifted - ref_shift_dist
                shifted = shift
                #! without if statement, parallel=True can cause a SystemError!
                # * parallel jit cannot handle changes in array dimension.
                # * without this, allele can change from a 1D array to a 0D
                # * array.
                if allele_start_idx == v_len:
                    # consume track up to end of variant
                    track_idx = v_rel_end
                    continue
                # consume track up to start of variant
                track_idx = v_rel_pos
                # adjust variant length
                v_len -= allele_start_idx

        # SNPs (but not MNPs because we don't have ALT length, MNPs are not atomic)
        # skipped because for tracks they always match the reference
        if v_diff == 0:
            continue

        # add track values up to variant
        track_len = v_rel_pos - track_idx
        if out_idx + track_len >= length:
            # track will get written by final clause
            # handles case where extraneous variants downstream of the haplotype were provided
            break
        out[out_idx : out_idx + track_len] = track[track_idx : track_idx + track_len]
        out_idx += track_len

        # indels (substitutions are skipped above and then handled by above clause)
        writable_length = min(v_len, length - out_idx)
        if v_diff > 0 and strategy_id != _REPEAT_5P:
            _apply_insertion_fill(
                out=out,
                out_idx=out_idx,
                writable_length=writable_length,
                v_len=v_len,
                track=track,
                v_rel_pos=v_rel_pos,
                strategy_id=strategy_id,
                params=params,
                base_seed=base_seed,
                query=query,
                hap=hap,
            )
        else:
            # Deletions and Repeat5p insertions: original behavior.
            for i in range(writable_length):
                out[out_idx + i] = track[v_rel_pos]
        out_idx += writable_length
        track_idx = v_rel_end

        if out_idx >= length:
            break

    if shifted < shift:
        # need to shift the rest of the track
        track_idx += shift - shifted
        track_idx = min(track_idx, len(track))
        shifted = shift

    # fill rest with track and pad with 0
    unfilled_length = length - out_idx
    if unfilled_length > 0:
        writable_ref = min(unfilled_length, len(track) - track_idx)
        out_end_idx = out_idx + writable_ref
        ref_end_idx = track_idx + writable_ref
        out[out_idx:out_end_idx] = track[track_idx:ref_end_idx]

        if out_end_idx < length:
            out[out_end_idx:] = 0


# -----------------------------------------------------------------------------
# Ragged helper: stack (batch, None) Rageds along a new track axis -> (batch, n_tracks, None)
# -----------------------------------------------------------------------------


def _ragged_stack_tracks(tracks: "list[Ragged]") -> "Ragged":
    """Stack *n_tracks* ``(batch, None)`` Rageds into a single ``(batch, n_tracks, None)`` Ragged.

    Each input has canonical 1-D offsets after ``.to_packed()``.  We interleave the segments
    so that for region ``r`` the output has n_tracks consecutive ragged rows (one per track).

    Implementation is fully vectorized — no Python loop over batch elements and no
    per-element ``np.concatenate``.  The only loop is a bounded ``n_tracks`` pass to
    scatter each track's segments into the pre-allocated output buffer.

    Algorithm
    ---------
    1. Stack the per-track 1-D offsets into an ``(n_tracks, n_batch+1)`` matrix and derive
       a ``(n_tracks, n_batch)`` lengths matrix.
    2. Transpose to ``(n_batch, n_tracks)`` and flatten → interleaved lengths in C-order,
       which is exactly the ``ak.concatenate(axis=1)`` segment order we need to reproduce.
    3. Convert to offsets (cumsum) → ``out_offsets`` of length ``n_batch*n_tracks + 1``.
    4. Allocate ``out_data`` of size ``total_elements`` in one shot.
    5. For each track ``t`` (bounded loop over *n_tracks*, never over *n_batch*):
       - Compute the destination start positions for all ``n_batch`` segments of that
         track using the pre-computed ``out_offsets`` (index ``t, t+n_tracks, t+2*n_tracks, …``).
       - Compute the source lengths for those segments from the stacked lengths matrix.
       - Walk the batch dimension with a vectorised numpy repeat/arange gather, writing
         all of track ``t``'s elements into ``out_data`` in a single pass via a boolean
         mask or explicit index array — both O(total_data), no per-element call.
    """
    from seqpro.rag._core import Ragged as _CoreRagged, RaggedLayout

    if not tracks:
        raise ValueError("_ragged_stack_tracks: empty track list")

    n_tracks = len(tracks)
    n_batch = int(tracks[0].offsets.shape[0]) - 1

    if n_tracks == 1:
        # Single track: just promote (batch, None) -> (batch, 1, None) by
        # re-labelling the shape — no data copy needed.
        t = tracks[0]
        new_layout = RaggedLayout(
            data=t._rl.data,
            offsets=list(t._layout.offsets),
            shape=(n_batch, 1, None),
            str_offsets=t._rl.str_offsets,
        )
        return _CoreRagged(new_layout)

    # ------------------------------------------------------------------
    # 1. Vectorized lengths: (n_tracks, n_batch)
    # ------------------------------------------------------------------
    # Stack all per-track offsets into one matrix (n_tracks, n_batch+1).
    # np.diff over axis=1 gives lengths (n_tracks, n_batch).
    all_offsets = np.stack([t.offsets for t in tracks], axis=0)  # (n_tracks, n_batch+1)
    lengths_tk = np.diff(all_offsets, axis=1)                     # (n_tracks, n_batch)

    # ------------------------------------------------------------------
    # 2. Interleaved lengths (n_batch, n_tracks) → flat, then offsets
    # ------------------------------------------------------------------
    # Transposing to (n_batch, n_tracks) and flattening gives the
    # segment order [batch0_track0, batch0_track1, …, batchN_track(T-1)],
    # which exactly matches ak.concatenate(axis=1) semantics.
    out_lengths = lengths_tk.T.ravel()                            # (n_batch * n_tracks,)
    out_offsets = lengths_to_offsets(out_lengths)                 # (n_batch*n_tracks + 1,)
    total = int(out_offsets[-1])

    # ------------------------------------------------------------------
    # 3. Allocate output buffer once
    # ------------------------------------------------------------------
    out_data = np.empty(total, dtype=tracks[0].data.dtype)

    # ------------------------------------------------------------------
    # 4. Scatter each track's data — loop is O(n_tracks), NOT O(n_batch)
    # ------------------------------------------------------------------
    # For track t, its segments land at flat indices t, t+n_tracks, …
    # out_offsets[t::n_tracks] are the destination starts for all n_batch
    # segments of track t.  We build a flat index array by repeating each
    # destination start length[r] times, giving us one output slot per
    # source element — fully vectorized via np.repeat + np.arange.
    for t_idx, t in enumerate(tracks):
        dst_starts = out_offsets[t_idx::n_tracks][:n_batch]  # (n_batch,) destination starts
        seg_lens   = lengths_tk[t_idx]                        # (n_batch,) lengths

        # Build flat output indices for every element of track t_idx:
        # for segment r starting at dst_starts[r] with length seg_lens[r],
        # the output positions are dst_starts[r], dst_starts[r]+1, …
        # np.repeat(dst_starts, seg_lens) gives the base, and
        # np.arange over the cumulative offsets gives the per-element delta.
        seg_offsets_src = all_offsets[t_idx, :n_batch]       # source starts in t.data
        src_data_total  = int(seg_lens.sum())
        if src_data_total == 0:
            continue

        # Intra-segment offset (0,1,2,…,len-1 repeated per segment)
        intra = np.arange(src_data_total, dtype=np.int64)
        intra -= np.repeat(
            np.concatenate(([0], seg_lens[:-1].cumsum())), seg_lens
        )

        dst_idx = np.repeat(dst_starts, seg_lens) + intra
        src_idx = np.repeat(seg_offsets_src, seg_lens) + intra

        out_data[dst_idx] = t.data[src_idx]

    return _CoreRagged.from_offsets(out_data, (n_batch, n_tracks, None), out_offsets)


# -----------------------------------------------------------------------------
# Tracks reconstructor (Python-level wrapper around the numba kernels above).
# -----------------------------------------------------------------------------


class TrackType(enum.Enum):
    SAMPLE = enum.auto()
    ANNOT = enum.auto()


_T = TypeVar("_T", RaggedTracks, RaggedIntervals)
_NewT = TypeVar("_NewT", RaggedTracks, RaggedIntervals)


@dataclass(slots=True)
class Tracks(Reconstructor[_T]):
    intervals: dict[str, RaggedIntervals]
    """The intervals in the dataset. This is memory mapped."""
    active_tracks: dict[str, TrackType]
    available_tracks: dict[str, TrackType]
    kind: type[_T]
    n_regions: int
    n_samples: int
    insertion_fill: dict[str, InsertionFill] = field(default_factory=dict)
    """Per-track insertion fill strategy. Defaults to Repeat5p for every active track."""

    def with_tracks(self, tracks: str | Iterable[str] | None) -> Tracks:
        if tracks is None:
            return replace(self, active_tracks={}, insertion_fill={})

        if isinstance(tracks, str):
            _tracks = [tracks]
        else:
            _tracks = tracks

        if missing := list(set(_tracks) - set(self.intervals)):
            raise ValueError(f"Missing tracks: {missing}")

        tracks = {t: self.available_tracks[t] for t in _tracks}
        fills = {t: self.insertion_fill.get(t, Repeat5p()) for t in _tracks}
        return replace(self, active_tracks=tracks, insertion_fill=fills)

    def with_insertion_fill(
        self,
        fill: InsertionFill | Mapping[str, InsertionFill],
    ) -> Tracks:
        """Configure the insertion-fill strategy for each active track.

        Parameters
        ----------
        fill
            Either a single :class:`InsertionFill` strategy applied to every
            active track, or a mapping from track name to strategy. Track names
            not present in the mapping fall back to :class:`Repeat5p`.
        """
        if isinstance(fill, InsertionFill):
            fills = {name: fill for name in self.active_tracks}
        else:
            fills = {name: fill.get(name, Repeat5p()) for name in self.active_tracks}
        return replace(self, insertion_fill=fills)

    @classmethod
    def from_path(
        cls,
        path: Path,
        n_regions: int,
        n_samples: int,
        kind: type[_T] = RaggedTracks,
    ) -> Tracks[_T]:
        strack_dir = path / "intervals"
        atrack_dir = path / "annot_intervals"

        def _is_track_dir(p: Path) -> bool:
            return (
                p.is_dir()
                and ".tmp." not in p.name
                and ".old." not in p.name
                and not p.name.endswith(".lock")
            )

        available_tracks: list[str] = []
        if strack_dir.exists():
            for p in strack_dir.iterdir():
                if not _is_track_dir(p):
                    continue
                if len(list(p.iterdir())) == 0:
                    p.rmdir()
                else:
                    available_tracks.append(p.name)
            available_tracks.sort()

        available_annots: list[str] = []
        if atrack_dir.exists():
            for p in atrack_dir.iterdir():
                if not _is_track_dir(p):
                    continue
                if len(list(p.iterdir())) == 0:
                    p.rmdir()
                else:
                    available_annots.append(p.name)
            available_annots.sort()

        if name_clash := set(available_tracks) & set(available_annots):
            raise ValueError(
                f"Found sample and annotation tracks with the same name: {name_clash}"
            )

        intervals: dict[str, RaggedIntervals] | None = {}

        for track in available_tracks:
            intervals[track] = cls._open_intervals(
                strack_dir / track, n_regions, n_samples
            )

        for track in available_annots:
            intervals[track] = cls._open_intervals(atrack_dir / track, n_regions, 0)

        all_tracks = dict(
            zip(available_tracks, itertools.repeat(TrackType.SAMPLE))
        ) | dict(zip(available_annots, itertools.repeat(TrackType.ANNOT)))

        insertion_fill = {name: Repeat5p() for name in all_tracks}
        return cls(
            intervals,
            all_tracks,
            all_tracks,
            kind,
            n_regions,
            n_samples,
            insertion_fill,
        )

    @staticmethod
    def _open_intervals(path: Path, n_regions: int, n_samples: int) -> RaggedIntervals:
        if n_samples == 0:
            shape = (n_regions, None)
        else:
            shape = (n_regions, n_samples, None)
        itvs = np.memmap(
            path / "intervals.npy",
            dtype=INTERVAL_DTYPE,
            mode="r",
        )
        offsets = np.memmap(
            path / "offsets.npy",
            dtype=np.int64,
            mode="r",
        )
        starts = Ragged.from_offsets(itvs["start"], shape, offsets)
        ends = Ragged.from_offsets(itvs["end"], shape, offsets)
        values = Ragged.from_offsets(itvs["value"], shape, offsets)
        return RaggedIntervals(starts, ends, values)

    def to_kind(self, kind: type[_NewT]) -> Tracks[_NewT]:
        t = replace(self, kind=kind)
        return cast(Tracks[_NewT], t)

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: SplicePlan | None = None,
        flat: bool = False,
    ) -> _T:
        if splice_plan is not None and not issubclass(self.kind, RaggedTracks):
            raise NotImplementedError(
                "Splicing of RaggedIntervals tracks is not supported."
            )
        if issubclass(self.kind, RaggedTracks):
            out = self._call_float32(
                idx, r_idx, regions, output_length, splice_plan=splice_plan
            )
        else:
            out = self._call_intervals(idx, flat=flat)
        return cast(_T, out)

    def _call_float32(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        splice_plan: SplicePlan | None = None,
    ) -> RaggedTracks:
        batch_size = len(idx)

        if isinstance(output_length, int):
            out_lengths = track_lengths = np.full(batch_size, output_length)
        else:
            lengths = regions[:, 2] - regions[:, 1]
            out_lengths = track_lengths = lengths

        if splice_plan is None:
            # (b [p])
            out_ofsts_per_t = lengths_to_offsets(out_lengths)
            track_ofsts_per_t = lengths_to_offsets(track_lengths)
            # caller accounts for ploidy
            n_per_track: int = out_ofsts_per_t[-1]
            # ragged (b t [p] l)
            out = np.empty(len(self.active_tracks) * n_per_track, np.float32)
            out_lens = repeat(out_lengths, "b -> b t", t=len(self.active_tracks))
            out_offsets = lengths_to_offsets(out_lens)

            for track_ofst, (name, tracktype) in enumerate(self.active_tracks.items()):
                intervals = self.intervals[name]
                # (b t l) ragged
                _out = out[track_ofst * n_per_track : (track_ofst + 1) * n_per_track]

                if tracktype is TrackType.SAMPLE:
                    o_idx = idx
                else:
                    o_idx = r_idx

                intervals_to_tracks(
                    offset_idxs=o_idx,
                    starts=regions[:, 1],
                    itv_starts=intervals.starts.data,
                    itv_ends=intervals.ends.data,
                    itv_values=intervals.values.data,
                    itv_offsets=intervals.starts.offsets,
                    out=_out,
                    out_offsets=track_ofsts_per_t,
                )

            out_shape = (len(idx), len(self.active_tracks), None)
            # flat (b t l)
            return cast(RaggedTracks, _Flat.from_offsets(out, out_shape, out_offsets))

        # ---- splice plan path ----
        assert not isinstance(output_length, int), (
            "splice plan path requires variable/ragged output"
        )
        # The plan was built with inner_fixed = (n_tracks,) so plan.permutation has
        # length B*T indexed in (query, track) C-order: k = query * T + track.
        # Each k_new in the permuted order targets one (query, track) pair; we
        # need to write its bytes into out_buf at plan.permuted_out_offsets[k_new].
        n_tracks = len(self.active_tracks)
        total = int(splice_plan.permuted_out_offsets[-1])
        out_buf = np.empty(total, np.float32)

        k_old = splice_plan.permutation  # length B*T
        track_of_k = k_old % n_tracks
        query_of_k = k_old // n_tracks

        for track_ofst, (name, tracktype) in enumerate(self.active_tracks.items()):
            mask = track_of_k == track_ofst
            if not mask.any():
                continue
            # k_new indices that target this track, in permuted order.
            k_new_idx = np.flatnonzero(mask)
            queries = query_of_k[k_new_idx]  # length M
            intervals = self.intervals[name]
            o_idx_full = idx if tracktype is TrackType.SAMPLE else r_idx
            sub_lengths = regions[queries, 2] - regions[queries, 1]
            sub_offsets = lengths_to_offsets(sub_lengths)
            scratch = np.empty(int(sub_offsets[-1]), np.float32)
            intervals_to_tracks(
                offset_idxs=o_idx_full[queries],
                starts=regions[queries, 1],
                itv_starts=intervals.starts.data,
                itv_ends=intervals.ends.data,
                itv_values=intervals.values.data,
                itv_offsets=intervals.starts.offsets,
                out=scratch,
                out_offsets=sub_offsets,
            )
            # Scatter scratch[m] into out_buf at the global permuted position.
            perm_out = splice_plan.permuted_out_offsets
            for m, k_new in enumerate(k_new_idx):
                s_dest = int(perm_out[k_new])
                e_dest = int(perm_out[k_new + 1])
                s_src = int(sub_offsets[m])
                e_src = int(sub_offsets[m + 1])
                out_buf[s_dest:e_dest] = scratch[s_src:e_src]

        # Per-element flat (caller rewraps with group_offsets via _regroup).
        out_shape = (splice_plan.permuted_lengths.shape[0], None)
        return cast(
            RaggedTracks,
            _Flat.from_offsets(out_buf, out_shape, splice_plan.permuted_out_offsets),
        )

    def _call_intervals(
        self, idx: NDArray[np.integer], flat: bool = False
    ) -> RaggedIntervals | FlatIntervals:
        r_idx, s_idx = np.unravel_index(idx, (self.n_regions, self.n_samples))

        if flat:
            return build_flat_intervals(
                self.active_tracks, self.intervals, r_idx, s_idx, self.n_samples
            )

        # out = (batch tracks ~itvs)
        # Collect per-track (batch, None) Rageds, then interleave into (batch, n_tracks, None).
        per_track_starts: list[Ragged] = []
        per_track_ends: list[Ragged] = []
        per_track_values: list[Ragged] = []

        for name, tracktype in self.active_tracks.items():
            # (regions [samples] ~itvs)
            intervals = self.intervals[name]
            if tracktype is TrackType.SAMPLE:
                # (batch ~itvs)
                itvs = intervals[r_idx, s_idx].to_packed()
            else:
                # (batch ~itvs)
                itvs = intervals[r_idx].to_packed()
            per_track_starts.append(itvs.starts)
            per_track_ends.append(itvs.ends)
            per_track_values.append(itvs.values)

        # (batch tracks ~itvs) by interleaving n_tracks (batch, None) Rageds
        starts = _ragged_stack_tracks(per_track_starts)
        ends = _ragged_stack_tracks(per_track_ends)
        values = _ragged_stack_tracks(per_track_values)
        return RaggedIntervals(starts, ends, values)

    def write_transformed_track(
        self,
        new_track: str,
        existing_track: str,
        transform: Callable[
            [NDArray[np.intp], NDArray[np.intp], Ragged[np.float32]],
            Ragged[np.float32],
        ],
        path: Path,
        regions: NDArray[np.int32],
        max_jitter: int,
        idxer: DatasetIndexer,
        haps: Haps | None = None,
        max_mem: int = 2**30,
        overwrite: bool = False,
    ) -> Tracks:
        # The pre-Awkward-Ragged implementation lived here as a chunked
        # decompress -> user transform -> recompress loop; it did not survive
        # the migration of Ragged to its Awkward-backed form. See
        # `docs/superpowers/roadmap.md` for what would be needed to revive it,
        # and `git show 1f1b718:python/genvarloader/_dataset/_reconstruct.py`
        # for the previous implementation.
        raise NotImplementedError(
            "write_transformed_track is not implemented for the current "
            "Awkward-backed Ragged. See docs/superpowers/roadmap.md "
            '("Transformed track writing") for the revival plan.'
        )


def build_flat_intervals(
    active_tracks: dict[str, TrackType],
    intervals: dict[str, RaggedIntervals],
    r_idx: NDArray[np.integer],
    s_idx: NDArray[np.integer],
    n_samples: int,
) -> FlatIntervals:
    """Pure-numpy gather of per-(region, sample, track) intervals into a
    :class:`FlatIntervals` of shape ``(batch, n_tracks, ~itvs)`` in C-order
    (batch outer, track inner) — matching the awkward concat order of
    :meth:`Tracks._call_intervals`.
    """
    B = len(r_idx)
    T = len(active_tracks)

    # Pass 1: gather each track's B groups in batch order (t, b layout).
    tb_starts: list[NDArray] = []
    tb_ends: list[NDArray] = []
    tb_values: list[NDArray] = []
    lengths_tb = np.empty((T, B), np.int64)
    for t, (name, tracktype) in enumerate(active_tracks.items()):
        itv = intervals[name]
        if tracktype is TrackType.SAMPLE:
            g = r_idx * n_samples + s_idx
        else:
            g = r_idx
        off = np.asarray(itv.starts.offsets)
        lo = off[g]
        lens = (off[g + 1] - lo).astype(np.int64)
        lengths_tb[t] = lens
        pt_off = lengths_to_offsets(lens)
        total = int(pt_off[-1])
        src = np.repeat(lo - pt_off[:-1], lens) + np.arange(total, dtype=np.int64)
        tb_starts.append(np.asarray(itv.starts.data)[src])
        tb_ends.append(np.asarray(itv.ends.data)[src])
        tb_values.append(np.asarray(itv.values.data)[src])

    data_starts = np.concatenate(tb_starts) if tb_starts else np.empty(0, np.int32)
    data_ends = np.concatenate(tb_ends) if tb_ends else np.empty(0, np.int32)
    data_values = np.concatenate(tb_values) if tb_values else np.empty(0, np.float32)
    offsets_tb = lengths_to_offsets(lengths_tb.ravel())  # (T*B + 1)

    # Pass 2: reorder groups (t, b) -> (b, t). For output group (b, t) the
    # source group in (t, b) layout is t*B + b.
    perm = (np.arange(T)[None, :] * B + np.arange(B)[:, None]).ravel()  # (B*T,)
    final_lengths = lengths_tb.ravel()[perm]
    final_offsets = lengths_to_offsets(final_lengths)
    total = int(final_offsets[-1])
    src = np.repeat(offsets_tb[perm] - final_offsets[:-1], final_lengths) + np.arange(
        total, dtype=np.int64
    )

    shape = (B, T, None)
    return FlatIntervals(
        starts=_Flat.from_offsets(data_starts[src], shape, final_offsets),
        ends=_Flat.from_offsets(data_ends[src], shape, final_offsets),
        values=_Flat.from_offsets(data_values[src], shape, final_offsets),
    )
