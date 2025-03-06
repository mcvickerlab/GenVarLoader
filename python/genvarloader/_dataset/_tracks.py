from typing import Optional

import numba as nb
import numpy as np
from numpy.typing import NDArray

__all__ = []


@nb.njit(parallel=True, nogil=True, cache=True)
def shift_and_realign_tracks(
    regions: NDArray[np.int32],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    first_v_idxs: NDArray[np.uint32],
    offsets: NDArray[np.uint32],
    genos: NDArray[np.int8],
    shifts: NDArray[np.uint32],
    tracks: NDArray[np.float32],
    out: NDArray[np.float32],
):
    """Shift and realign tracks to correspond to haplotypes.

    Parameters
    ----------
    regions : NDArray[np.int32]
        Shape = (n_regions, 3) Regions, each is (contig_idx, start, end).
    positions : NDArray[np.int32]
        Shape = (n_variants) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (n_variants) Sizes of variants.
    genos : NDArray[np.int8]
        Shape = (n_variants, ploidy) Genotypes of variants.
    shifts : NDArray[np.uint32]
        Shape = (n_regions, ploidy) Shifts for each haplotype.
    first_v_idxs : NDArray[np.uint32]
        Shape = (n_regions) Index of first variant for each region.
    offsets : NDArray[np.uint32]
        Shape = (n_regions + 1) Offsets into genos.
    tracks : NDArray[np.uint32]
        Shape = (n_regions, length) Tracks.
    out : NDArray[np.float32]
        Shape = (ploidy, length) Shifted and re-aligned tracks.
    """
    n_regions = len(first_v_idxs)
    ploidy = genos.shape[1]
    length = out.shape[2]
    for query in nb.prange(n_regions):
        _out = out[query]
        query_s = regions[query, 1]
        _shifts = shifts[query]

        _track = tracks[query]

        o_s, o_e = offsets[query], offsets[query + 1]
        n_variants = o_e - o_s

        if n_variants == 0:
            _out[:] = _track[:length]
            continue

        _genos = genos[o_s:o_e]

        v_s = first_v_idxs[query]
        v_e = v_s + n_variants
        # adjust positions to be relative to track subsequence
        _positions = positions[v_s:v_e] - query_s
        _sizes = sizes[v_s:v_e]

        for hap in nb.prange(ploidy):
            shift_and_realign_track(
                _positions,
                _sizes,
                _genos[:, hap],
                _shifts[hap],
                _track,
                _out[hap],
            )


@nb.njit(nogil=True, cache=True)
def shift_and_realign_track(
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    genos: NDArray[np.int8],
    shift: int,
    track: NDArray[np.float32],
    out: NDArray[np.float32],
):
    """Shift and realign a track to correspond to a haplotype.

    Parameters
    ----------
    positions : NDArray[np.int32]
        Shape = (n_variants) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (n_variants) Sizes of variants.
    genos : NDArray[np.int8]
        Shape = (n_variants) Genotypes of variants.
    shift : int
        Total amount to shift by.
    track : NDArray[np.float32]
        Shape = (length) Track.
    out : NDArray[np.uint8]
        Shape = (out_length) Shifted and re-aligned track.
    """
    length = len(out)
    n_variants = len(positions)
    # where to get next reference subsequence
    track_idx = 0
    # where to put next subsequence
    out_idx = 0
    # how much we've shifted
    shifted = 0

    # first variant is a DEL spanning start
    v_rel_pos = positions[0]
    v_diff = sizes[0]
    if v_rel_pos < 0 and genos[0] == 1:
        # diff of v(-1) has been normalized to consider where ref is
        # otherwise, ref_idx = v_rel_pos - v_diff + 1
        # e.g. a -10 diff became -3 if v_rel_pos = -7
        track_idx = v_rel_pos - v_diff + 1
        # increment the variant index
        start_idx = 1
    else:
        start_idx = 0

    for variant in range(start_idx, n_variants):
        # UNKNOWN -9 or REF 0
        if genos[variant] != 1:
            continue

        # position of variant relative to ref from fetch(contig, start, q_end)
        # i.e. has been put into same coordinate system as ref_idx
        v_rel_pos = positions[variant]

        # overlapping variants
        # v_rel_pos < ref_idx only if we see an ALT at a given position a second
        # time or more. We'll do what bcftools consensus does and only use the
        # first ALT variant we find.
        if v_rel_pos < track_idx:
            continue

        v_diff = sizes[variant]
        v_len = max(0, v_diff + 1)
        value = track[..., v_rel_pos]

        # handle shift
        if shifted < shift:
            ref_shift_dist = v_rel_pos - track_idx
            # not enough distance to finish the shift even with the variant
            if shifted + ref_shift_dist + v_len < shift:
                # consume ref up to the end of the variant
                track_idx = v_rel_pos + 1
                # add the length of skipped ref and size of the variant to the shift
                shifted += ref_shift_dist + v_len
                # skip the variant
                continue
            # enough distance between ref_idx and variant to finish shift
            elif shifted + ref_shift_dist >= shift:
                track_idx += shift - shifted
                shifted = shift
                # can still use the variant and whatever ref is left between
                # ref_idx and the variant
            # ref + (some of) variant is enough to finish shift
            else:
                # consume ref up to beginning of variant
                # ref_idx will be moved to end of variant after using the variant
                track_idx = v_rel_pos
                # how much left to shift - amount of ref we can use
                allele_start_idx = shift - shifted - ref_shift_dist
                #! without if statement, parallel=True can cause a SystemError!
                # * parallel jit cannot handle changes in array dimension.
                # * without this, allele can change from a 1D array to a 0D
                # * array.
                if allele_start_idx == v_len:
                    continue
                v_len -= allele_start_idx
                # done shifting
                shifted = shift

        # add track values up to variant
        track_len = v_rel_pos - track_idx
        if out_idx + track_len >= length:
            # track will get written by final clause
            # handles case where extraneous variants downstream of the haplotype were provided
            break
        out[out_idx : out_idx + track_len] = track[track_idx : track_idx + track_len]
        out_idx += track_len

        # insertions + substitions
        writable_length = min(v_len, length - out_idx)
        out[out_idx : out_idx + writable_length] = value
        out_idx += writable_length
        # +1 because ALT alleles always replace 1 nt of reference for a
        # normalized VCF
        track_idx = v_rel_pos + 1

        # deletions, move ref to end of deletion
        if v_diff < 0:
            track_idx -= v_diff

        if out_idx >= length:
            break

    # fill rest with track and pad with 0
    unfilled_length = length - out_idx
    if unfilled_length > 0:
        writable_ref = min(unfilled_length, len(track) - track_idx)
        out_end_idx = out_idx + writable_ref
        ref_end_idx = track_idx + writable_ref
        out[out_idx:out_end_idx] = track[track_idx:ref_end_idx]

        if out_end_idx < length:
            out[out_end_idx:] = 0


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
    offset_idx : NDArray[np.intp]
        Shape = (regions, ploidy) Indices into offsets for each region.
    variant_idxs : NDArray[np.int32]
        Shape = (variants) Indices of variants.
    offsets : NDArray[np.uint32]
        Shape = (samples*ploidy*total_regions + 1) Offsets into variant idxs.
    regions : NDArray[np.int32]
        Shape = (n_regions, 3) Regions, each is (contig_idx, start, end).
    positions : NDArray[np.int32]
        Shape = (total_variants) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (total_variants) Sizes of variants.
    shifts : NDArray[np.int32]
        Shape = (regions, ploidy) Shifts for each haplotype.
    tracks : NDArray[np.float32]
        Shape = (total_length) Tracks.
    track_offsets : NDArray[np.int64]
        Shape = (regions + 1) Offsets into tracks.
    out : NDArray[np.float32]
        Ragged array with shape (regions, ploidy). Shifted and re-aligned tracks.
    out_offsets : NDArray[np.int64]
        Shape = (regions*ploidy + 1) Offsets into out.
    keep : Optional[NDArray[np.bool_]]
        Shape = (variants) Keep mask for genotypes.
    keep_offsets : Optional[NDArray[np.int64]]
        Shape = (regions*ploidy + 1) Offsets into keep.
    """
    n_regions, ploidy = geno_offset_idxs.shape
    for query in nb.prange(n_regions):
        t_s, t_e = track_offsets[query], track_offsets[query + 1]
        q_track = tracks[t_s:t_e]
        # assumes start is never altered upstream by differing hap lengths
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
    variant_idxs : NDArray[np.int32]
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
            # not enough distance to finish the shift even with the variant
            if shifted + ref_shift_dist + v_len < shift:
                # consume ref up to the end of the variant
                track_idx = v_rel_end
                # add the length of skipped ref and size of the variant to the shift
                shifted += ref_shift_dist + v_len
                # skip the variant
                continue
            # enough distance between ref_idx and variant to finish shift
            elif shifted + ref_shift_dist >= shift:
                shifted = shift
                track_idx += shift - shifted
                # can still use the variant and whatever ref is left between
                # ref_idx and the variant
            # ref + (some of) variant is enough to finish shift
            else:
                shifted = shift
                # how much left to shift - amount of ref we can use
                allele_start_idx = shift - shifted - ref_shift_dist
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

    # fill rest with track and pad with 0
    unfilled_length = length - out_idx
    if unfilled_length > 0:
        writable_ref = min(unfilled_length, len(track) - track_idx)
        out_end_idx = out_idx + writable_ref
        ref_end_idx = track_idx + writable_ref
        out[out_idx:out_end_idx] = track[track_idx:ref_end_idx]

        if out_end_idx < length:
            out[out_end_idx:] = 0
