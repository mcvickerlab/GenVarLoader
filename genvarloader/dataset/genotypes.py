from typing import Tuple

import numba as nb
import numpy as np
from attrs import define
from numpy.typing import NDArray

from ..types import ListIdx
from ..utils import lengths_to_offsets
from .utils import padded_slice


@define
class DenseGenotypes:
    """Dense genotypes. In this format, genotypes are stored as a special case of a ragged 3D array where
    each sample has the same number of variants, but each region may have a different number of variants.
    Thus, the first variant indices are the same for every sample, and the offsets are readily computed
    from the first sample's offsets given the number of samples. The genotypes are laid out in C order such
    that the first `n_variant` rows are the genotypes for the first sample, the next `n_variant` rows are
    the genotypes for the second sample, and so on.

    Attributes
    ----------
    genos : NDArray[np.int8]
        Shape = (n_samples * n_variants, ploidy) Genotypes.
    first_v_idxs : NDArray[np.uint32]
        Shape = (n_regions,) First variant index for each region.
    offsets : NDArray[np.uint32]
        Shape = (n_regions + 1,) Offsets into genos.
    n_samples : int
        Number of samples.
    """

    genos: NDArray[np.int8]  # (n_samples * n_variants, ploidy)
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

    def __getitem__(self, idx: Tuple[ListIdx, ListIdx]) -> "DenseGenotypes":
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

        return DenseGenotypes(genos, first_v_idxs, offsets, self.n_samples)


@nb.njit(parallel=True, nogil=True, cache=True)
def first_v_idxs_to_all_v_idxs(first_variant_indices, n_per_region):
    """Convert first variant indices to variant indices."""
    out = np.empty(n_per_region.sum(), dtype=np.int32)
    out_start = np.empty_like(n_per_region)
    out_start[0] = 0
    out_start[1:] = n_per_region[:-1].cumsum()
    for i in nb.prange(len(first_variant_indices)):
        _f = first_variant_indices[i]
        _n = n_per_region[i]
        if _n == 0:
            continue
        _o_s = out_start[i]
        out[_o_s : _o_s + _n] = np.arange(_f, _f + _n, dtype=np.int32)
    return out


@define
class SparseGenotypes:
    """Sparse genotypes corresponding to distinct regions. In this format, genotypes are stored as a ragged 3D array where each
    sample, ploid, and region may have a different number of variants, since unknown and REF genotypes are not stored. The
    variant indices are aligned to the genotypes. Physically, the genotypes and variant indices are stored as 1D arrays.
    Then, each sample and region's info can be sliced out using the offsets:
    >>> i = np.ravel_multi_index((s, p, r), (n_samples, ploidy, n_regions))
    >>> genos[offsets[i]:offsets[i+1]]
    >>> variant_idxs[offsets[i]:offsets[i+1]]

    Attributes
    ----------
    genos : NDArray[np.int8]
        Shape = (samples * ploidy * variants) Genotypes.
    variant_idxs : NDArray[np.int32]
        Shape = (samples * ploidy * variants) Variant indices.
    offsets : NDArray[np.int32]
        Shape = (samples * ploidy * regions + 1) Offsets into genos.
    n_samples : int
        Number of samples.
    ploidy : int
        Ploidy.
    n_regions : int
        Number of regions.
    """

    variant_idxs: NDArray[np.int32]  # (samples * ploidy * variants)
    offsets: NDArray[np.int32]  # (samples * ploidy * regions + 1)
    n_samples: int
    ploidy: int
    n_regions: int

    @property
    def effective_shape(self):
        return (self.n_samples, self.ploidy, self.n_regions)

    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_regions: int):
        """Create an empty sparse genotypes object."""
        return cls(
            np.empty(0, np.int32),
            np.zeros(n_samples * ploidy * n_regions + 1, np.int32),
            n_samples,
            ploidy,
            n_regions,
        )

    def vars(self, sample: int, ploidy: int, region: int):
        """Get variant indices for a given sample and region."""
        i = np.ravel_multi_index(
            (sample, ploidy, region), (self.n_samples, self.ploidy, self.n_regions)
        )
        vars = self.variant_idxs[self.offsets[i] : self.offsets[i + 1]]
        return vars

    @classmethod
    def from_dense(
        cls,
        genos: NDArray[np.int8],
        first_v_idxs: NDArray[np.int32],
        offsets: NDArray[np.int32],
    ):
        """Convert dense genotypes to sparse genotypes.

        Parameters
        ----------
        genos : NDArray[np.int8]
            Shape = (sample, ploidy, variants) Genotypes.
        first_v_idxs : NDArray[np.uint32]
            Shape = (regions) First variant index for each region.
        offsets : NDArray[np.uint32]
            Shape = (regions + 1) Offsets into genos.
        """
        n_per_region = np.diff(offsets)
        dense_v_idxs = first_v_idxs_to_all_v_idxs(first_v_idxs, n_per_region)
        # (s p v)
        is_alt = genos == 1
        alt = is_alt.nonzero()
        variant_idxs = dense_v_idxs[alt[2]]
        n_per_spr = np.add.reduceat(is_alt, offsets[:-1], axis=-1).ravel()
        offsets = lengths_to_offsets(n_per_spr, np.int32)
        return cls(
            variant_idxs,
            offsets,
            genos.shape[0],
            genos.shape[1],
            len(first_v_idxs),
        )

    @classmethod
    def from_dense_with_length(
        cls,
        genos: NDArray[np.int8],
        first_v_idxs: NDArray[np.int32],
        offsets: NDArray[np.int32],
        ilens: NDArray[np.int32],
        positions: NDArray[np.int32],
        starts: NDArray[np.int32],
        length: int,
    ):
        """Convert dense genotypes to sparse genotypes.

        Parameters
        ----------
        genos : NDArray[np.int8]
            Shape = (sample, ploidy, variants) Genotypes.
        first_v_idxs : NDArray[np.uint32]
            Shape = (regions) First variant index for each region.
        offsets : NDArray[np.uint32]
            Shape = (regions + 1) Offsets into genos.
        ilens : NDArray[np.int32]
            Shape = (variants) ILEN of all unique variants.
        length : int
            Length of the output haplotypes.
        """
        n_samples = genos.shape[0]
        ploidy = genos.shape[1]
        n_regions = len(first_v_idxs)
        n_per_region = np.diff(offsets)
        # (v)
        dense_v_idxs = first_v_idxs_to_all_v_idxs(first_v_idxs, n_per_region)
        # (s p v)
        spv_ilens = np.where(genos == 1, ilens[dense_v_idxs], 0)
        # (s p v)
        cum_ilens = spv_ilens.cumsum(-1)
        # (s p r)
        cum_r_ilens = np.add.reduceat(spv_ilens, offsets[:-1], axis=-1).cumsum(-1)
        del spv_ilens
        # (s p v)
        keep = keep_genotypes(
            genos,
            cum_ilens,
            cum_r_ilens,
            offsets,
            first_v_idxs,
            positions,
            starts,
            length,
        )
        keep_idxs = keep.nonzero()
        variant_idxs = dense_v_idxs[keep_idxs[2]]
        n_per_spr = np.add.reduceat(keep, offsets[:-1], axis=-1).ravel()
        offsets = lengths_to_offsets(n_per_spr, np.int32)
        # (r)
        largest_v_idxs_per_region = variant_idxs[
            offsets[1:].reshape(n_samples, ploidy, n_regions) - 1
        ].max((0, 1))
        max_ends = positions[largest_v_idxs_per_region] - ilens[
            largest_v_idxs_per_region
        ].clip(max=0)
        sparse_genos = cls(
            variant_idxs,
            offsets,
            genos.shape[0],
            genos.shape[1],
            len(first_v_idxs),
        )
        return sparse_genos, max_ends


@nb.njit(parallel=True, nogil=True, cache=True)
def keep_genotypes(
    genos: NDArray[np.int8],
    cum_ilens: NDArray[np.int32],
    cum_r_ilens: NDArray[np.int32],
    offsets: NDArray[np.int32],
    first_v_idxs: NDArray[np.int32],
    positions: NDArray[np.int32],
    starts: NDArray[np.int32],
    length: int,
):
    """Will mark genotypes to keep based on being an ALT allele and being within the length of the haplotype.

    Parameters
    ----------
    genos : NDArray[np.int8]
        Shape = (n_samples, ploidy, n_variants) Genotypes.
    cum_ilens : NDArray[np.int32]
        Shape = (n_samples, ploidy, n_variants) Cumulative lengths of haplotypes.
    cum_r_ilens : NDArray[np.int32]
        Shape = (n_samples, ploidy, n_regions) Cumulative lengths of regions.
    offsets : NDArray[np.int32]
        Shape = (n_regions + 1) Offsets into genos.
    first_v_idxs : NDArray[np.int32]
        Shape = (n_regions,) First variant index for each region.
    positions : NDArray[np.int32]
        Shape = (total_variants,) Positions of variants.
    starts : NDArray[np.int32]
        Shape = (n_regions,) Start of query regions.
    length : int
        Length of haplotypes.
    """
    n_samples = cum_ilens.shape[0]
    ploidy = cum_ilens.shape[1]
    n_regions = len(starts)
    keep = np.empty_like(cum_ilens, np.bool_)
    for sample in nb.prange(n_samples):
        for ploid in nb.prange(ploidy):
            for region in nb.prange(n_regions):
                o_s, o_e = offsets[region], offsets[region + 1]
                n_variants = o_e - o_s
                if n_variants == 0:
                    continue
                r_start = starts[region]
                for variant in nb.prange(o_s, o_e):
                    v_idx = first_v_idxs[region] + variant - o_s
                    rel_pos = positions[v_idx] - r_start
                    cum_ilen = (
                        cum_ilens[sample, ploid, variant]
                        - cum_r_ilens[sample, ploid, region]
                    )
                    if (
                        rel_pos + cum_ilen < length
                        and genos[sample, ploid, variant] == 1
                    ):
                        keep[sample, ploid, variant] = True
                    else:
                        keep[sample, ploid, variant] = False
    return keep


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
def get_diffs_sparse(
    offset_idxs: NDArray[np.intp],
    sparse_genos: NDArray[np.int32],
    offsets: NDArray[np.int32],
    size_diffs: NDArray[np.int32],
):
    """Get difference in length wrt reference genome for given genotypes.

    Parameters
    ----------
    offset_idxs : NDArray[np.intp]
        Shape = (n_regions, ploidy) Indices for each region into offsets.
    sparse_genos : NDArray[np.int32]
        Shape = (samples*ploidy*variants) Sparse genotypes i.e. variant indices for ALT genotypes.
    offsets : NDArray[np.int32]
        Shape = (samples*ploidy*regions + 1) Offsets into sparse genotypes.
    size_diffs : NDArray[np.int32]
        Shape = (total_variants) Size of all unique variants.
    """
    n_queries, ploidy = offset_idxs.shape
    diffs = np.empty(offset_idxs.shape, np.int32)
    for query in nb.prange(n_queries):
        for ploid in nb.prange(ploidy):
            o_idx = offset_idxs[query, ploid]
            o_s, o_e = offsets[o_idx], offsets[o_idx + 1]
            n_variants = o_e - o_s
            if n_variants == 0:
                diffs[query, ploid] = 0
            else:
                v_idxs = sparse_genos[o_s:o_e]
                diffs[query, ploid] = size_diffs[v_idxs].sum()
    return diffs


@nb.njit(parallel=True, nogil=True, cache=True)
def reconstruct_haplotypes_from_dense(
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
    """Reconstruct haplotypes from reference sequence and variants.

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
            reconstruct_haplotype_from_dense(
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
def reconstruct_haplotype_from_dense(
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


@nb.njit(parallel=True, nogil=True, cache=True)
def reconstruct_haplotypes_from_sparse(
    offset_idxs: NDArray[np.intp],
    out: NDArray[np.uint8],
    regions: NDArray[np.int32],
    shifts: NDArray[np.int32],
    offsets: NDArray[np.int32],
    sparse_genos: NDArray[np.int32],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    alt_alleles: NDArray[np.uint8],
    alt_offsets: NDArray[np.uintp],
    ref: NDArray[np.uint8],
    ref_offsets: NDArray[np.uint64],
    pad_char: int,
):
    """Reconstruct haplotypes from reference sequence and variants.

    Parameters
    ----------
    offset_idxs: NDArray[np.intp]
        Shape = (n_regions, ploidy) Indices for each region into offsets.
    out : NDArray[np.uint8]
        Shape = (n_regions, ploidy, out_length) Output array.
    regions : NDArray[np.int32]
        Shape = (n_regions, 3) Regions to reconstruct haplotypes.
    shifts : NDArray[np.uint32]
        Shape = (n_regions, ploidy) Shifts for each region.
    offsets : NDArray[np.uint32]
        Shape = (ploidy*n_regions + 1) Offsets into genos.
    sparse_genos : NDArray[np.int32]
        Shape = (variants) Sparse genotypes of variants i.e. variant indices for ALT genotypes.
    positions : NDArray[np.int32]
        Shape = (total_variants) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (total_variants) Sizes of variants.
    alt_alleles : NDArray[np.uint8]
        Shape = (total_alt_length) ALT alleles.
    alt_offsets : NDArray[np.uintp]
        Shape = (total_variants) Offsets of ALT alleles.
    ref : NDArray[np.uint8]
        Shape = (ref_length) Reference sequence.
    ref_offsets : NDArray[np.uint64]
        Shape = (n_contigs) Offsets of reference sequences.
    pad_char : int
        Padding character.
    n_samples : int
        Number of samples.
    ploidy : int
        Ploidy.
    n_regions : int
        Number of regions.
    """
    n_regions = out.shape[0]
    ploidy = out.shape[1]
    for query in nb.prange(n_regions):
        q = regions[query]
        c_idx = q[0]
        c_s = ref_offsets[c_idx]
        c_e = ref_offsets[c_idx + 1]
        ref_s = q[1]
        ref_e = q[2]
        _ref = padded_slice(ref[c_s:c_e], ref_s, ref_e, pad_char)

        for ploid in nb.prange(ploidy):
            o_idx = offset_idxs[query, ploid]
            _out = out[query, ploidy]
            _shifts = shifts[query, ploid]

            reconstruct_haplotype_from_sparse(
                o_idx,
                sparse_genos,
                offsets,
                positions,
                sizes,
                _shifts,
                alt_alleles,
                alt_offsets,
                _ref,
                ref_s,
                _out,
                pad_char,
            )


@nb.njit(nogil=True, cache=True)
def reconstruct_haplotype_from_sparse(
    offset_idx: int,
    variant_idxs: NDArray[np.int32],
    offsets: NDArray[np.int32],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    shift: int,
    alt_alleles: NDArray[np.uint8],  # full set
    alt_offsets: NDArray[np.uintp],  # full set
    ref: NDArray[np.uint8],
    ref_start: int,
    out: NDArray[np.uint8],
    pad_char: int,
):
    """Reconstruct a haplotype from reference sequence and variants.

    Parameters
    ----------
    sample_idx : int
        Sample index.
    ploid : int
        Ploidy.
    region_idx : int
        Region index.
    variant_idxs : NDArray[np.int8]
        Shape = (variants) Genotypes of variants.
    offsets : NDArray[np.int32]
        Shape = (samples*ploidy*regions + 1) Offsets into variant indices.
    positions : NDArray[np.int32]
        Shape = (total_variants) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (total_variants) Sizes of variants.
    shift : int
        Shift amount.
    alt_alleles : NDArray[np.uint8]
        Shape = (total_alt_length) ALT alleles.
    alt_offsets : NDArray[np.uintp]
        Shape = (total_variants + 1) Offsets of ALT alleles.
    ref : NDArray[np.uint8]
        Shape = (ref_length) Reference sequence. ref_length >= out_length
    ref_start : int
        Start position of reference sequence.
    out : NDArray[np.uint8]
        Shape = (out_length) Output array.
    pad_char : int
        Padding character.
    """
    _variant_idxs = variant_idxs[offsets[offset_idx] : offsets[offset_idx + 1]]
    length = len(out)
    n_variants = len(_variant_idxs)
    if n_variants == 0:
        out[:] = ref[:length]
        return

    # where to get next reference subsequence
    ref_idx = 0
    # where to put next subsequence
    out_idx = 0
    # total amount to shift by
    shift = shift
    # how much we've shifted
    shifted = 0

    # first variant is a DEL spanning start
    v_rel_pos = positions[_variant_idxs[0]] - ref_start
    v_diff = sizes[_variant_idxs[0]]
    if v_rel_pos < 0:
        # diff of v(-1) has been normalized to consider where ref is
        # otherwise, ref_idx = v_rel_pos - v_diff + 1
        # e.g. a -10 diff became -3 if v_rel_pos = -7
        ref_idx = v_rel_pos - v_diff + 1
        # increment the variant index
        start_idx = 1
    else:
        start_idx = 0

    for v in range(start_idx, n_variants):
        variant: np.int32 = _variant_idxs[v]
        # position of variant relative to ref from fetch(contig, start, q_end)
        # i.e. has been put into same coordinate system as ref_idx
        v_rel_pos = positions[variant] - ref_start

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
