from typing import Sequence, Tuple, Union

import numba as nb
import numpy as np
from attrs import define
from numpy.typing import NDArray

from .utils import padded_slice

Idx = Union[int, np.integer, Sequence[int], NDArray[np.integer], slice]
ListIdx = Union[Sequence[int], NDArray[np.integer]]


@define
class Genotypes:
    genos: NDArray[np.int8]  # (n_variants * n_samples, ploidy)
    first_v_idxs: NDArray[np.uint32]  # (n_regions)
    offsets: NDArray[np.uint32]  # (n_regions + 1)
    n_samples: int

    @property
    def n_regions(self) -> int:
        return len(self.first_v_idxs)

    @property
    def n_variants(self) -> int:
        return len(self.genos) // self.n_samples

    def __len__(self) -> int:
        return len(self.first_v_idxs)

    def __getitem__(self, idx: Tuple[ListIdx, ListIdx]) -> "Genotypes":
        s_idx = idx[0]
        r_idx = idx[1]
        genos = []
        first_v_idxs = self.first_v_idxs[r_idx]
        offsets = np.empty(len(r_idx) + 1, dtype=np.uint32)
        offsets[0] = 0
        shifts = np.asarray(s_idx) * self.n_variants
        for output_idx, (shift, region) in enumerate(zip(shifts, r_idx), 1):
            s, e = self.offsets[region] + shift, self.offsets[region + 1] + shift
            offsets[output_idx] = e - s
            if e > s:
                genos.append(self.genos[s:e])
        if len(genos) == 0:
            genos = np.empty((0, self.genos.shape[1]), dtype=self.genos.dtype)
        else:
            genos = np.concatenate(genos)
        offsets = offsets.cumsum(dtype=np.uint32)

        return Genotypes(genos, first_v_idxs, offsets, self.n_samples)


@nb.njit(parallel=True, nogil=True, cache=True)
def get_diffs(
    first_v_idxs: NDArray[np.uint32],
    offsets: NDArray[np.uint32],
    genotypes: NDArray[np.int8],
    size_diffs: NDArray[np.int32],
) -> NDArray[np.uint32]:
    """Get difference in length wrt reference genome for given genotypes.

    Parameters
    ----------
    first_v_idxs : NDArray[np.uint32]
        Shape = (n_regions,) First variant index for each query.
    offsets : NDArray[np.uint32]
        Shape = (n_regions + 1,) Offsets into genos.
    genotypes : NDArray[np.int8]
        Shape = (n_variants, ploidy) Genotypes.
    size_diffs : NDArray[np.int32]
        Shape = (total_variants,) Size of variants.
    """
    n_regions = len(first_v_idxs)
    ploidy = genotypes.shape[1]
    diffs = np.empty((n_regions, ploidy), np.uint32)

    for region in nb.prange(n_regions):
        o_s, o_e = offsets[region], offsets[region + 1]
        n_variants = o_e - o_s

        if n_variants == 0:
            diffs[region] = 0
            continue

        v_s = first_v_idxs[region]
        v_e = v_s + n_variants
        # (v p)
        genos = genotypes[o_s:o_e]
        # (v p) -> (p)
        diff = np.where(genos == 1, size_diffs[v_s:v_e, None], 0).sum(0).clip(0)
        diffs[region] = diff
    return diffs


@nb.njit(parallel=True, nogil=True, cache=True)
def reconstruct_haplotypes(
    out: NDArray[np.uint8],
    regions: NDArray[np.int32],
    shifts: NDArray[np.uint32],
    first_v_idxs: NDArray[np.uint32],
    offsets: NDArray[np.uint32],
    genos: NDArray[np.int8],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    alt_alleles: NDArray[np.uint8],
    alt_offsets: NDArray[np.uintp],
    ref: NDArray[np.uint8],
    ref_offsets: NDArray[np.uint64],
    pad_char: int,
):
    """_summary_

    Parameters
    ----------
    out : NDArray[np.uint8]
        Shape = (n_regions, ploidy, out_length) Output array.
    regions : NDArray[np.int32]
        Shape = (n_regions, 3) Regions to reconstruct haplotypes.
    shifts : NDArray[np.uint32]
        Shape = (n_regions, ploidy) Shifts for each query.
    first_v_idxs : NDArray[np.uint32]
        Shape = (n_regions,) First variant index for each query.
    offsets : NDArray[np.uint32]
        Shape = (n_regions + 1,) Offsets into genos.
    genos : NDArray[np.int8]
        Shape = (n_variants, ploidy) Genotypes of variants.
    positions : NDArray[np.int32]
        Shape = (total_variants,) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (total_variants,) Sizes of variants.
    alt_alleles : NDArray[np.uint8]
        Shape = (total_alt_length,) ALT alleles.
    alt_offsets : NDArray[np.uintp]
        Shape = (total_variants,) Offsets of ALT alleles.
    ref : NDArray[np.uint8]
        Shape = (ref_length,) Reference sequence.
    ref_offsets : NDArray[np.uint64]
        Shape = (n_contigs,) Offsets of reference sequences.
    pad_char : int
        Padding character.
    """
    n_regions = len(first_v_idxs)
    ploidy = genos.shape[1]
    length = out.shape[2]
    for query in nb.prange(n_regions):
        _out = out[query]
        q = regions[query]
        _shifts = shifts[query]

        c_idx = q[0]
        c_s = ref_offsets[c_idx]
        c_e = ref_offsets[c_idx + 1]
        ref_s = q[1]
        ref_e = q[2]
        _ref = padded_slice(ref[c_s:c_e], ref_s, ref_e, pad_char)

        o_s, o_e = offsets[query], offsets[query + 1]
        n_variants = o_e - o_s

        if n_variants == 0:
            _out[:] = _ref[:length]
            continue

        _genos = genos[o_s:o_e]

        v_s = first_v_idxs[query]
        v_e = v_s + n_variants
        # adjust positions to be relative to reference subsequence
        _positions = positions[v_s:v_e] - ref_s
        _sizes = sizes[v_s:v_e]
        _alt_offsets = alt_offsets[v_s : v_e + 1].copy()
        _alt_alleles = alt_alleles[_alt_offsets[0] : _alt_offsets[-1]]
        _alt_offsets -= _alt_offsets[0]

        for hap in nb.prange(ploidy):
            reconstruct_haplotype(
                _positions,
                _sizes,
                _genos[:, hap],
                _shifts[hap],
                _alt_alleles,
                _alt_offsets,
                _ref,
                _out[hap],
                pad_char,
            )


@nb.njit(nogil=True, cache=True)
def reconstruct_haplotype(
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    genos: NDArray[np.int8],
    shift: int,
    alt_alleles: NDArray[np.uint8],
    alt_offsets: NDArray[np.uintp],
    ref: NDArray[np.uint8],
    out: NDArray[np.uint8],
    pad_char: int,
):
    """Reconstruct a haplotype from reference sequence and variants.

    Parameters
    ----------
    positions : NDArray[np.int32]
        Shape = (n_variants) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (n_variants) Sizes of variants.
    genos : NDArray[np.int8]
        Shape = (n_variants) Genotypes of variants.
    shift : int
        Shift amount.
    alt_alleles : NDArray[np.uint8]
        Shape = (total_alt_length) ALT alleles.
    alt_offsets : NDArray[np.uintp]
        Shape = (n_variants) Offsets of ALT alleles.
    ref : NDArray[np.uint8]
        Shape = (ref_length) Reference sequence. ref_length >= out_length
    out : NDArray[np.uint8]
        Shape = (out_length) Output array.
    pad_char : int
        Padding character.
    """
    length = len(out)
    n_variants = len(positions)
    # where to get next reference subsequence
    ref_idx = 0
    # where to put next subsequence
    out_idx = 0
    # total amount to shift by
    shift = shift
    # how much we've shifted
    shifted = 0

    # first variant is a DEL spanning start
    v_rel_pos = positions[0]
    v_diff = sizes[0]
    if v_rel_pos < 0 and genos[0] == 1:
        # diff of v(-1) has been normalized to consider where ref is
        # otherwise, ref_idx = v_rel_pos - v_diff + 1
        # e.g. a -10 diff became -3 if v_rel_pos = -7
        ref_idx = v_rel_pos - v_diff + 1
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
        if v_rel_pos < ref_idx:
            continue

        v_diff = sizes[variant]
        allele = alt_alleles[alt_offsets[variant] : alt_offsets[variant + 1]]
        v_len = len(allele)

        # handle shift
        if shifted < shift:
            ref_shift_dist = v_rel_pos - ref_idx
            # not enough distance to finish the shift even with the variant
            if shifted + ref_shift_dist + v_len < shift:
                # consume ref up to the end of the variant
                ref_idx = v_rel_pos + 1
                # add the length of skipped ref and size of the variant to the shift
                shifted += ref_shift_dist + v_len
                # skip the variant
                continue
            # enough distance between ref_idx and variant to finish shift
            elif shifted + ref_shift_dist >= shift:
                ref_idx += shift - shifted
                shifted = shift
                # can still use the variant and whatever ref is left between
                # ref_idx and the variant
            # ref + (some of) variant is enough to finish shift
            else:
                # consume ref up to beginning of variant
                # ref_idx will be moved to end of variant after using the variant
                ref_idx = v_rel_pos
                # how much left to shift - amount of ref we can use
                allele_start_idx = shift - shifted - ref_shift_dist
                #! without if statement, parallel=True can cause a SystemError!
                # * parallel jit cannot handle changes in array dimension.
                # * without this, allele can change from a 1D array to a 0D
                # * array.
                if allele_start_idx == v_len:
                    continue
                allele = allele[allele_start_idx:]
                v_len = len(allele)
                # done shifting
                shifted = shift

        # add reference sequence
        ref_len = v_rel_pos - ref_idx
        if out_idx + ref_len >= length:
            # ref will get written by final clause
            # handles case where extraneous variants downstream of the haplotype were provided
            break
        out[out_idx : out_idx + ref_len] = ref[ref_idx : ref_idx + ref_len]
        out_idx += ref_len

        # insertions + substitions
        writable_length = min(v_len, length - out_idx)
        out[out_idx : out_idx + writable_length] = allele[:writable_length]
        out_idx += writable_length
        # +1 because ALT alleles always replace 1 nt of reference for a
        # normalized VCF
        ref_idx = v_rel_pos + 1

        # deletions, move ref to end of deletion
        if v_diff < 0:
            ref_idx -= v_diff

        if out_idx >= length:
            break

    # fill rest with reference sequence and pad with Ns
    unfilled_length = length - out_idx
    if unfilled_length > 0:
        writable_ref = min(unfilled_length, len(ref) - ref_idx)
        out_end_idx = out_idx + writable_ref
        ref_end_idx = ref_idx + writable_ref
        out[out_idx:out_end_idx] = ref[ref_idx:ref_end_idx]

        if out_end_idx < length:
            out[out_end_idx:] = pad_char
