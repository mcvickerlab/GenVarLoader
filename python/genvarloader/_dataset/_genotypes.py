import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(parallel=True, nogil=True, cache=True)
def get_diffs_sparse(
    geno_offset_idxs: NDArray[np.integer],
    geno_v_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    ilens: NDArray[np.integer],
    keep: NDArray[np.bool_] | None = None,
    keep_offsets: NDArray[np.integer] | None = None,
    q_starts: NDArray[np.integer] | None = None,
    q_ends: NDArray[np.integer] | None = None,
    v_starts: NDArray[np.integer] | None = None,
):
    """Get difference in length wrt reference genome for given genotypes.

    If starts, ends, & positions are given, they take priority over keep and keep_offsets.

    Parameters
    ----------
    geno_offset_idxs : NDArray[np.intp]
        Shape = (n_regions, ploidy) Indices for each region into offsets.
    geno_v_idxs : NDArray[np.int32]
        Shape = (variants*samples*ploidy) Sparse genotypes i.e. variant indices for ALT genotypes.
    geno_offsets : NDArray[np.int32]
        Shape = (regions*samples*ploidy + 1) Offsets into sparse genotypes.
    ilens : NDArray[np.int32]
        Shape = (total_variants) Size of all unique variants.
    keep : Optional[NDArray[np.bool_]]
        Shape = (variants*samples*ploidy) Keep mask for genotypes.
    keep_offsets : Optional[NDArray[np.int64]]
        Shape = (regions*samples*ploidy + 1) Offsets into keep.
    q_starts : Optional[NDArray[np.int32]]
        Shape = (regions) Start of query regions.
    q_ends : Optional[NDArray[np.int32]]
        Shape = (regions) End of query regions.
    v_starts : Optional[NDArray[np.int32]]
        Shape = (total_variants) Positions of unique variants.
    """
    n_queries, ploidy = geno_offset_idxs.shape
    diffs = np.empty((n_queries, ploidy), np.int32)
    for query in nb.prange(n_queries):
        for hap in nb.prange(ploidy):
            o_idx = geno_offset_idxs[query, hap]
            if geno_offsets.ndim == 1:
                o_s, o_e = geno_offsets[o_idx], geno_offsets[o_idx + 1]
            else:
                o_s, o_e = geno_offsets[o_idx]
            n_variants = o_e - o_s
            if n_variants == 0:
                diffs[query, hap] = 0
            elif q_starts is not None and q_ends is not None and v_starts is not None:
                diffs[query, hap] = 0
                for v in range(o_s, o_e):
                    if keep is not None and keep_offsets is not None:
                        k_s = keep_offsets[query * ploidy + hap]
                        v_keep = keep[k_s + (v - o_s)]
                        if not v_keep:
                            continue

                    v_idx: int = geno_v_idxs[v]
                    v_start = v_starts[v_idx]
                    v_ilen = ilens[v_idx]
                    # +1 assumes atomized variants
                    v_end = v_start - min(0, v_ilen) + 1

                    if v_end <= q_starts[query]:
                        # variant doesn't span region
                        continue

                    if v_start >= q_ends[query]:
                        # variants are sorted by position so this variant and everything
                        # after will be outside the region
                        break

                    # deletion may start before region
                    #     0 1 2 3 4 5 6
                    # DEL s - - r e - - : +max(0, 3 - 0) -> -3 + 3 = 0
                    # DEL r - s - e - - : +max(0, 0 - 2) -> -1 + 0 = -1
                    # where r is region start, s is variant start, e is variant end (exclusive)
                    # count the "-" to get ilen
                    # but also atomic deletions include 1 bp of ref so add it back (- 1)
                    if v_ilen < 0:
                        v_ilen += max(0, q_starts[query] - v_start - 1)
                    # deletion may end after region
                    v_ilen += max(0, v_end - q_ends[query])

                    diffs[query, hap] += v_ilen
            elif keep is not None and keep_offsets is not None:
                v_idxs = geno_v_idxs[o_s:o_e]
                k_idx = query * ploidy + hap
                qh_keep = keep[keep_offsets[k_idx] : keep_offsets[k_idx + 1]]
                v_idxs = v_idxs[qh_keep]
                diffs[query, hap] = ilens[v_idxs].sum()
            else:
                diffs[query, hap] = ilens[geno_v_idxs[o_s:o_e]].sum()
    return diffs


@nb.njit(parallel=True, nogil=True, cache=True)
def reconstruct_haplotypes_from_sparse(
    out: NDArray[np.uint8],
    out_offsets: NDArray[np.integer],
    regions: NDArray[np.integer],
    shifts: NDArray[np.integer],
    geno_offset_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    geno_v_idxs: NDArray[np.integer],
    v_starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    alt_alleles: NDArray[np.uint8],
    alt_offsets: NDArray[np.integer],
    ref: NDArray[np.uint8],
    ref_offsets: NDArray[np.integer],
    pad_char: int,
    keep: NDArray[np.bool_] | None = None,
    keep_offsets: NDArray[np.int64] | None = None,
    annot_v_idxs: NDArray[np.int32] | None = None,
    annot_ref_pos: NDArray[np.int32] | None = None,
):
    """Reconstruct haplotypes from reference sequence and variants.

    Parameters
    ----------
    offset_idxs: NDArray[np.intp]
        Shape = (n_regions, ploidy) Indices for each region into offsets.
    out : NDArray[np.uint8]
        Ragged array of shape = (n_regions, ploidy) to write haplotypes into.
    out_offsets : NDArray[np.int64]
        Shape = (n_regions*ploidy + 1) Offsets into out.
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
    keep : Optional[NDArray[np.bool_]]
        Shape = (variants) Keep mask for genotypes.
    annot_v_idxs : Optional[NDArray[np.int32]]
        Ragged array of shape (n_regions, ploidy). Variant indices for annotations.
    annot_ref_pos : Optional[NDArray[np.int32]]
        Ragged array of shape (n_regions, ploidy). Reference positions for annotations.
    """
    n_regions, ploidy = geno_offset_idxs.shape
    for query in nb.prange(n_regions):
        q = regions[query]
        c_idx: int = q[0]
        c_s = ref_offsets[c_idx]
        c_e = ref_offsets[c_idx + 1]
        ref_start: int = q[1]
        _reference = ref[c_s:c_e]

        for hap in nb.prange(ploidy):
            # index for full sparse genos
            o_idx = geno_offset_idxs[query, hap]

            # local index for subset of variants that are implied by offset_idxs
            k_idx = query * ploidy + hap
            if keep is not None and keep_offsets is not None:
                qh_keep = keep[keep_offsets[k_idx] : keep_offsets[k_idx + 1]]
            else:
                qh_keep = None

            # aligned to out sequence
            out_s, out_e = out_offsets[k_idx], out_offsets[k_idx + 1]
            qh_out = out[out_s:out_e]
            qh_shift = shifts[query, hap]

            qh_annot_v_idxs = (
                annot_v_idxs[out_s:out_e] if annot_v_idxs is not None else None
            )
            qh_annot_ref_pos = (
                annot_ref_pos[out_s:out_e] if annot_ref_pos is not None else None
            )

            reconstruct_haplotype_from_sparse(
                offset_idx=o_idx,
                geno_v_idxs=geno_v_idxs,
                geno_offsets=geno_offsets,
                v_starts=v_starts,
                ilens=ilens,
                shift=qh_shift,
                alt_alleles=alt_alleles,
                alt_offsets=alt_offsets,
                ref=_reference,
                ref_start=ref_start,
                out=qh_out,
                pad_char=pad_char,
                keep=qh_keep,
                annot_v_idxs=qh_annot_v_idxs,
                annot_ref_pos=qh_annot_ref_pos,
            )


@nb.njit(nogil=True, cache=True)
def reconstruct_haplotype_from_sparse(
    offset_idx: int,
    geno_v_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    v_starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    shift: int,
    alt_alleles: NDArray[np.uint8],  # full set
    alt_offsets: NDArray[np.integer],  # full set
    ref: NDArray[np.uint8],  # full contig
    ref_start: int,  # may be negative
    out: NDArray[np.uint8],
    pad_char: int,
    keep: NDArray[np.bool_] | None = None,
    annot_v_idxs: NDArray[np.integer] | None = None,
    annot_ref_pos: NDArray[np.integer] | None = None,
):
    """Reconstruct a haplotype from reference sequence and variants.

    Parameters
    ----------
    offset_idx : int
        Index for `offsets` for where to find the offsets into variant_idxs.
    variant_idxs : int
        Index of alt variants for all samples and variants.
    offsets : NDArray[np.int32]
        Shape = Offsets into variant indices.
    positions : NDArray[np.int32]
        Shape = (total_variants) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (total_variants) Sizes of variants.
    shift : int
        Total amount to shift by.
    alt_alleles : NDArray[np.uint8]
        Shape = (total_alt_length) ALT alleles.
    alt_offsets : NDArray[np.uintp]
        Shape = (total_variants + 1) Offsets of ALT alleles.
    ref : NDArray[np.uint8]
        Shape = (ref_length) Reference sequence for the whole contig. ref_length >= out_length
    ref_start : int
        Start position of reference sequence, may be negative.
    out : NDArray[np.uint8]
        Shape = (out_length) Output array.
    pad_char : int
        Padding character.
    keep: Optional[NDArray[np.bool_]]
        Shape = (variants) Keep mask for genotypes.
    annot_v_idxs: Optional[NDArray[np.int32]]
        Shape = (out_length) Variant indices for annotations.
    annot_ref_pos: Optional[NDArray[np.int32]]
        Shape = (out_length) Reference positions for annotations
    """
    if geno_offsets.ndim == 1:
        o_s, o_e = geno_offsets[offset_idx], geno_offsets[offset_idx + 1]
    else:
        o_s, o_e = geno_offsets[offset_idx]
    _variant_idxs = geno_v_idxs[o_s:o_e]
    length = len(out)
    n_variants = len(_variant_idxs)

    # where to get next reference subsequence
    ref_idx = ref_start
    # where to put next subsequence
    out_idx = 0
    # how much we've shifted
    shifted = 0

    # if ref_idx is negative, we need to pad the beginning of the haplotype
    if ref_idx < 0:
        pad_len = -ref_idx
        shifted = min(shift, pad_len)
        pad_len -= shifted
        out[out_idx : out_idx + pad_len] = pad_char
        if annot_v_idxs is not None:
            annot_v_idxs[out_idx : out_idx + pad_len] = -1
        if annot_ref_pos is not None:
            annot_ref_pos[out_idx : out_idx + pad_len] = -1
        out_idx += pad_len
        ref_idx = 0

    for v in range(n_variants):
        if keep is not None and not keep[v]:
            continue

        variant: np.int32 = _variant_idxs[v]
        v_pos = v_starts[variant]
        v_diff = ilens[variant]
        allele = alt_alleles[alt_offsets[variant] : alt_offsets[variant + 1]]
        v_len = len(allele)
        # +1 assumes atomized variants, exactly 1 nt shared between REF and ALT
        v_ref_end = v_pos - min(0, v_diff) + 1

        # if variant is a DEL spanning start of query
        if v_pos < ref_start and v_diff < 0 and v_ref_end >= ref_start:
            ref_idx = v_ref_end
            continue

        # overlapping variants
        # v_rel_pos < ref_idx only if we see an ALT at a given position a second
        # time or more. We'll do what bcftools consensus does and only use the
        # first ALT variant we find.
        if v_pos < ref_idx:
            continue

        # handle shift
        if shifted < shift:
            ref_shift_dist = v_pos - ref_idx
            # not enough distance to finish the shift even with the variant
            if shifted + ref_shift_dist + v_len < shift:
                # skip the variant
                continue
            # enough distance between ref_idx and start of variant to finish shift
            elif shifted + ref_shift_dist >= shift:
                ref_idx += shift - shifted
                shifted = shift
                # can still use the variant and whatever ref is left between
                # ref_idx and the variant
            # ref + all or some of variant is enough to finish shift
            else:
                # how much left to shift - amount of ref we can use
                allele_start_idx = shift - shifted - ref_shift_dist
                shifted = shift
                #! without if statement, parallel=True can cause a SystemError!
                # * parallel jit cannot handle changes in array dimension.
                # * without this, allele can change from a 1D array to a 0D
                # * array.
                # enough dist with variant to complete shift
                if allele_start_idx == v_len:
                    # move ref to end of variant
                    ref_idx = v_ref_end
                    # skip the variant
                    continue
                # consume ref up to beginning of variant
                # ref_idx will be moved to end of variant after using the variant
                ref_idx = v_pos
                # adjust variant to start at allele_start_idx
                allele = allele[allele_start_idx:]
                v_len = len(allele)

        # add reference sequence
        ref_len = v_pos - ref_idx
        if out_idx + ref_len >= length:
            # ref will get written by final clause
            # handles case where extraneous variants downstream of the haplotype were provided
            break
        out[out_idx : out_idx + ref_len] = ref[ref_idx : ref_idx + ref_len]
        if annot_v_idxs is not None:
            annot_v_idxs[out_idx : out_idx + ref_len] = -1
        if annot_ref_pos is not None:
            annot_ref_pos[out_idx : out_idx + ref_len] = np.arange(
                ref_idx, ref_idx + ref_len
            )
        out_idx += ref_len

        # apply variant
        writable_length = min(v_len, length - out_idx)
        out[out_idx : out_idx + writable_length] = allele[:writable_length]
        if annot_v_idxs is not None:
            annot_v_idxs[out_idx : out_idx + writable_length] = variant
        if annot_ref_pos is not None:
            annot_ref_pos[out_idx : out_idx + writable_length] = v_pos
        out_idx += writable_length

        # advance ref_idx to end of variant
        ref_idx = v_ref_end

        if out_idx >= length:
            break

    if shifted < shift:
        # need to shift the rest of the track
        ref_idx += shift - shifted
        ref_idx = min(ref_idx, len(ref))
        shifted = shift

    # fill rest with reference sequence and right-pad with Ns
    unfilled_length = length - out_idx
    if unfilled_length > 0:
        # fill with reference sequence
        writable_ref = min(unfilled_length, len(ref) - ref_idx)
        out_end_idx = out_idx + writable_ref
        ref_end_idx = ref_idx + writable_ref
        out[out_idx:out_end_idx] = ref[ref_idx:ref_end_idx]
        if annot_v_idxs is not None:
            annot_v_idxs[out_idx:out_end_idx] = -1
        if annot_ref_pos is not None:
            annot_ref_pos[out_idx:out_end_idx] = np.arange(ref_idx, ref_end_idx)

        # right-pad
        if out_end_idx < length:
            out[out_end_idx:] = pad_char
            if annot_v_idxs is not None:
                annot_v_idxs[out_end_idx:] = -1
            if annot_ref_pos is not None:
                annot_ref_pos[out_end_idx:] = np.iinfo(np.int32).max
