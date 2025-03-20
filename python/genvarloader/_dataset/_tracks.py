from typing import Optional

import numba as nb
import numpy as np
from numpy.typing import NDArray

__all__ = []


@nb.njit(parallel=True, nogil=True, cache=True)
def shift_and_realign_tracks_sparse(
    out: NDArray[np.float32],
    out_offsets: NDArray[np.int64],
    regions: NDArray[np.int32],
    shifts: NDArray[np.int32],
    geno_offset_idxs: NDArray[np.integer],
    geno_v_idxs: NDArray[np.int32],
    geno_offsets: NDArray[np.int64],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    tracks: NDArray[np.float32],
    track_offsets: NDArray[np.int64],
    keep: Optional[NDArray[np.bool_]] = None,
    keep_offsets: Optional[NDArray[np.int64]] = None,
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
                positions=positions,
                sizes=sizes,
                shift=qh_shifts,
                track=q_track,
                query_start=q_start,
                out=qh_out,
                keep=qh_keep,
            )


@nb.njit(nogil=True, cache=True)
def shift_and_realign_track_sparse(
    offset_idx: int,
    geno_v_idxs: NDArray[np.int32],
    geno_offsets: NDArray[np.int64],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    shift: int,
    track: NDArray[np.float32],
    query_start: int,
    out: NDArray[np.float32],
    keep: Optional[NDArray[np.bool_]] = None,
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
    _variant_idxs = geno_v_idxs[geno_offsets[offset_idx] : geno_offsets[offset_idx + 1]]
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
        v_rel_pos = positions[variant] - query_start
        v_diff = sizes[variant]
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
        out[out_idx : out_idx + writable_length] = track[v_rel_pos]
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
