import json
from pathlib import Path
from typing import Dict, Union

import numba as nb
import numpy as np
from attrs import define
from numpy.typing import NDArray

from ..utils import lengths_to_offsets


@define
class GenomeTrack:
    """
    Represents a genome-wide track at base-pair resolution.

    Attributes
    ----------
    track : ndarray
        Track data.
    contigs : dict[str, int]
        A dictionary mapping contig names to their length.
    contig_offsets : ndarray
        The offsets for each contig in the genome track data.
    """

    track: NDArray
    contigs: Dict[str, int]
    contig_offsets: NDArray[np.int32]
    raw_contig_offsets: NDArray[np.int32]

    def order_offsets_like(self, contigs: list[str]):
        common, _, order = np.intersect1d(
            contigs, list(self.contigs.keys()), return_indices=True
        )
        if len(common) != len(contigs):
            raise ValueError("Not all contigs requested are in the track.")
        self.contig_offsets = self.raw_contig_offsets[order]
        return self

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "GenomeTrack":
        """
        Create a GenomeTrack from the specified path.

        Parameters
        ----------
        path : Union[str, Path]
            The path to the track data.

        Returns
        -------
        track : GenomeTrack
            A GenomeTrack object.

        """
        path = Path(path)
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
            contigs = metadata["contigs"]
            non_length_shape = metadata["non_length_shape"]
            dtype = np.dtype(metadata["dtype"])

        contig_offsets = lengths_to_offsets(contigs.values())
        track = np.memmap(path / "track.npy", mode="r", dtype=dtype).reshape(
            *non_length_shape, -1
        )
        return cls(track, contigs, contig_offsets, contig_offsets)


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
    offset_idxs: NDArray[np.intp],
    variant_idxs: NDArray[np.int32],
    offsets: NDArray[np.int32],
    regions: NDArray[np.int32],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    shifts: NDArray[np.int32],
    tracks: NDArray[np.float32],
    track_offsets: NDArray[np.int32],
    out: NDArray[np.float32],
):
    """Shift and realign tracks to correspond to haplotypes.

    Parameters
    ----------
    offset_idxs : NDArray[np.intp]
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
    shifts : NDArray[np.uint32]
        Shape = (regions, ploidy) Shifts for each haplotype.
    tracks : NDArray[np.uint32]
        Shape = (total_length) Tracks.
    track_offsets : NDArray[np.int32]
        Shape = (regions + 1) Offsets into tracks.
    out : NDArray[np.float32]
        Shape = (regions, ploidy, length) Shifted and re-aligned tracks.
    """
    n_regions = len(regions)
    ploidy = out.shape[1]
    for query in nb.prange(n_regions):
        t_s, t_e = track_offsets[query], track_offsets[query + 1]
        _track = tracks[t_s:t_e]
        query_s = regions[query, 1]

        for hap in nb.prange(ploidy):
            o_idx = offset_idxs[query, hap]
            _out = out[query, hap]
            _shifts = shifts[query, hap]

            shift_and_realign_track_sparse(
                o_idx,
                variant_idxs,
                offsets,
                positions,
                sizes,
                _shifts,
                _track,
                query_s,
                _out,
            )


@nb.njit(nogil=True, cache=True)
def shift_and_realign_track_sparse(
    offset_idx: int,
    variant_idxs: NDArray[np.int32],
    offsets: NDArray[np.int32],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    shift: int,
    track: NDArray[np.float32],
    query_start: int,
    out: NDArray[np.float32],
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
    """
    _variant_idxs = variant_idxs[offsets[offset_idx] : offsets[offset_idx + 1]]
    length = len(out)
    n_variants = len(_variant_idxs)
    if n_variants == 0:
        out[:] = track[:length]
        return
    # where to get next reference subsequence
    track_idx = 0
    # where to put next subsequence
    out_idx = 0
    # how much we've shifted
    shifted = 0

    # first variant is a DEL spanning start
    v_rel_pos = positions[_variant_idxs[0]] - query_start
    v_diff = sizes[_variant_idxs[0]]
    if v_rel_pos < 0:
        # diff of v(-1) has been normalized to consider where ref is
        # otherwise, ref_idx = v_rel_pos - v_diff + 1
        # e.g. a -10 diff became -3 if v_rel_pos = -7
        track_idx = v_rel_pos - v_diff + 1
        # increment the variant index
        start_idx = 1
    else:
        start_idx = 0

    for v in range(start_idx, n_variants):
        variant: np.int32 = _variant_idxs[v]

        # position of variant relative to ref from fetch(contig, start, q_end)
        # i.e. has been put into same coordinate system as ref_idx
        v_rel_pos = positions[variant] - query_start

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
