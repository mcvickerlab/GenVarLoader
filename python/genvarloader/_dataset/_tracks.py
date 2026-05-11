import numba as nb
import numpy as np
from numpy.typing import NDArray

__all__ = []

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
            if ref_idx < 0:
                ref_idx = 0
            xs[j] = -float(j)
            ys[j] = track[ref_idx]
        for j in range(k):
            ref_idx = v_rel_pos + 1 + j
            if ref_idx > track_len - 1:
                ref_idx = track_len - 1
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
    geno_offset_idxs: NDArray[np.integer],
    geno_v_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    v_starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    tracks: NDArray[np.floating],
    track_offsets: NDArray[np.integer],
    keep: NDArray[np.bool_] | None = None,
    keep_offsets: NDArray[np.integer] | None = None,
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
    geno_offset_idxs : NDArray[np.intp]
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
    n_regions, ploidy = geno_offset_idxs.shape
    for query in nb.prange(n_regions):
        t_s, t_e = track_offsets[query], track_offsets[query + 1]
        q_track = tracks[t_s:t_e]
        # assumes start is never altered upstream by differing hap lengths (true for left-aligned variants)
        q_start = regions[query, 1]

        for hap in nb.prange(ploidy):
            o_idx = geno_offset_idxs[query, hap]

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
                keep=qh_keep,
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
    keep: NDArray[np.bool_] | None = None,
    strategy_id: int = 0,
    params: NDArray[np.float64] | None = None,
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
        if v_diff > 0 and strategy_id != _REPEAT_5P and params is not None:
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
