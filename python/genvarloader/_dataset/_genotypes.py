import random
from typing import Optional, Tuple

import numba as nb
import numpy as np
from attrs import define
from numpy.typing import NDArray

from .._types import ListIdx
from .._utils import _lengths_to_offsets

__all__ = []


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
    first_v_idxs : NDArray[np.int32]
        Shape = (n_regions,) First variant index for each region.
    offsets : NDArray[np.int64]
        Shape = (n_regions + 1,) Offsets into genos.
    n_samples : int
        Number of samples.
    """

    genos: NDArray[np.int8]  # (n_samples * n_variants, ploidy)
    first_v_idxs: NDArray[np.int32]  # (n_regions)
    offsets: NDArray[np.int64]  # (n_regions + 1)
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
def first_v_idxs_to_all_v_idxs(first_variant_indices: NDArray, n_per_region: NDArray):
    """Convert first variant indices to variant indices."""
    out = np.empty(n_per_region.sum(), dtype=np.int32)
    out_start = np.empty_like(n_per_region)
    out_start[0] = 0
    out_start[1:] = n_per_region[:-1].cumsum()
    for i in nb.prange(len(first_variant_indices)):
        f = first_variant_indices[i]
        n = n_per_region[i]
        if n == 0:
            continue
        o_s = out_start[i]
        out[o_s : o_s + n] = np.arange(f, f + n, dtype=np.int32)
    return out


@nb.njit(parallel=True, nogil=True, cache=True)
def get_haplotype_region_ilens(
    genos: NDArray[np.int8],
    first_v_idxs: NDArray[np.int32],
    offsets: NDArray[np.int64],
    ilens: NDArray[np.int32],
):
    n_regions = len(first_v_idxs)
    n_samples = genos.shape[0]
    ploidy = genos.shape[1]
    r_ilens = np.zeros((n_samples, ploidy, n_regions), np.int32)
    for r in nb.prange(n_regions):
        o_s, o_e = offsets[r], offsets[r + 1]
        n_v = o_e - o_s
        if n_v == 0:
            continue
        fvi = first_v_idxs[r]
        for s in nb.prange(n_samples):
            for p in nb.prange(ploidy):
                r_ilens[s, p, r] = np.where(
                    genos[s, p, o_s:o_e] == 1, ilens[fvi : fvi + n_v], 0
                ).sum()
    return r_ilens


@define
class SparseGenotypes:
    """Sparse genotypes corresponding to distinct regions. In this format, genotypes are stored as a ragged 3D array where each
    sample, ploid, and region may have a different number of variants, since unknown and REF genotypes are not stored. The
    variant indices are aligned to the genotypes. Physically, the genotypes and variant indices are stored as 1D arrays.
    Then, each sample and region's info can be sliced out using the offsets:
    >>> i = np.ravel_multi_index((r, p, s), (n_regions, ploidy, n_samples))
    >>> variant_idxs[offsets[i]:offsets[i+1]]

    Attributes
    ----------
    variant_idxs : NDArray[np.int32]
        Shape = (variants * samples * ploidy) Variant indices.
    offsets : NDArray[np.int32]
        Shape = (regions * samples * ploidy + 1) Offsets into genos.
    n_samples : int
        Number of samples.
    ploidy : int
        Ploidy.
    n_regions : int
        Number of regions.
    """

    variant_idxs: NDArray[np.int32]  # (variants * samples * ploidy)
    offsets: NDArray[np.int64]  # (regions * samples * ploidy + 1)
    n_regions: int
    n_samples: int
    ploidy: int
    dosage: Optional[NDArray[np.float32]] = None  # (variants * samples)

    @property
    def effective_shape(self):
        return (self.n_regions, self.n_samples, self.ploidy)

    @classmethod
    def empty(cls, n_regions: int, n_samples: int, ploidy: int):
        """Create an empty sparse genotypes object."""
        return cls(
            np.empty(0, np.int32),
            np.zeros(n_regions * n_samples * ploidy + 1, np.int64),
            n_regions,
            n_samples,
            ploidy,
        )

    @property
    def is_empty(self) -> bool:
        return len(self.variant_idxs) == 0

    def vars(self, region: int, sample: int, ploidy: int):
        """Get variant indices for a given sample and region."""
        i = np.ravel_multi_index(
            (region, sample, ploidy), (self.n_regions, self.n_samples, self.ploidy)
        )
        vars = self.variant_idxs[self.offsets[i] : self.offsets[i + 1]]
        return vars

    def concat(*genos: "SparseGenotypes") -> "SparseGenotypes":
        """Concatenate sparse genotypes."""

        if not all(g.n_samples == genos[0].n_samples for g in genos):
            raise ValueError("All genotypes must have the same number of samples.")
        if not all(g.ploidy == genos[0].ploidy for g in genos):
            raise ValueError("All genotypes must have the same ploidy.")

        total_n_regions = sum(g.n_regions for g in genos)
        variant_idxs = np.concatenate([g.variant_idxs for g in genos])
        offsets = _lengths_to_offsets(
            np.concatenate([np.diff(g.offsets) for g in genos])
        )
        return SparseGenotypes(
            variant_idxs=variant_idxs,
            offsets=offsets,
            n_regions=total_n_regions,
            n_samples=genos[0].n_samples,
            ploidy=genos[0].ploidy,
        )

    @classmethod
    def from_dense(
        cls,
        genos: NDArray[np.int8],
        first_v_idxs: NDArray[np.int32],
        offsets: NDArray[np.int64],
        dosages: Optional[NDArray[np.float32]] = None,
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
        dosages : Optional[NDArray[np.float32]]
            Shape = (sample, ploidy, variants) Dosages.
        """
        n_regions = len(first_v_idxs)
        n_samples = genos.shape[0]
        ploidy = genos.shape[1]
        # (s p v)
        keep = genos == 1
        n_per_rsp = get_n_per_rsp(keep, offsets, n_regions)
        sparse_offsets = _lengths_to_offsets(n_per_rsp.ravel(), np.int64)
        variant_idxs = keep_mask_to_rsp_v_idx(
            keep, first_v_idxs, offsets, sparse_offsets, n_regions, n_samples, ploidy
        )
        if dosages is not None:
            dosages = dosages[keep]
        return cls(
            variant_idxs=variant_idxs,
            offsets=sparse_offsets,
            n_regions=n_regions,
            n_samples=n_samples,
            ploidy=ploidy,
            dosage=dosages,
        )

    @classmethod
    def from_dense_with_length(
        cls,
        genos: NDArray[np.int8],
        first_v_idxs: NDArray[np.int32],
        offsets: NDArray[np.int64],
        ilens: NDArray[np.int32],
        positions: NDArray[np.int32],
        starts: NDArray[np.int32],
        lengths: NDArray[np.int32],
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
            Shape = (total_variants) ILEN of all unique variants.
        positions : NDArray[np.int32]
            Shape = (total_variants) Positions of unique variants.
        starts : NDArray[np.int32]
            Shape = (regions) Start of query regions.
        lengths : NDArray[np.int32]
            Shape = (regions) Lengths of the output haplotypes.
        """
        n_regions = len(first_v_idxs)
        n_samples = genos.shape[0]
        ploidy = genos.shape[1]
        # (s p v)
        keep, min_ilens = get_keep_mask_for_length(
            genos,
            offsets,
            first_v_idxs,
            positions,
            ilens,
            starts,
            lengths,
        )
        # (r)
        max_ends: NDArray[np.int32] = starts + lengths - min_ilens.clip(max=0)
        # (r s p)
        n_per_rsp = get_n_per_rsp(keep, offsets, n_regions)
        sparse_offsets = _lengths_to_offsets(n_per_rsp.ravel(), np.int64)
        variant_idxs = keep_mask_to_rsp_v_idx(
            keep, first_v_idxs, offsets, sparse_offsets, n_regions, n_samples, ploidy
        )
        sparse_genos = cls(
            variant_idxs=variant_idxs,
            offsets=sparse_offsets,
            n_regions=n_regions,
            n_samples=n_samples,
            ploidy=ploidy,
        )
        return sparse_genos, max_ends


@nb.njit(parallel=True, nogil=True, cache=True)
def get_keep_mask_for_length(
    genos: NDArray[np.int8],
    offsets: NDArray[np.int64],
    first_v_idxs: NDArray[np.int32],
    positions: NDArray[np.int32],
    ilens: NDArray[np.int32],
    starts: NDArray[np.int32],
    lengths: NDArray[np.int32],
):
    """Mark genotypes to keep based on being an ALT allele and being within the length of the haplotype.

    Parameters
    ----------
    genos : NDArray[np.int8]
        Shape = (samples, ploidy, variants) Genotypes.
    offsets : NDArray[np.int32]
        Shape = (regions + 1) Offsets into genos.
    first_v_idxs : NDArray[np.int32]
        Shape = (regions) First variant index for each region.
    positions : NDArray[np.int32]
        Shape = (total_variants) Positions of variants.
    ilens : NDArray[np.int32]
        Shape = (total_variants) ILEN of all unique variants.
    starts : NDArray[np.int32]
        Shape = (regions) Start of query regions.
    lengths : NDArray[np.int32]
        Shape = (regions) Length of haplotypes.
    """
    n_samples = genos.shape[0]
    ploidy = genos.shape[1]
    n_regions = len(starts)
    keep = np.empty_like(genos, np.bool_)
    out_ilens = np.zeros((n_regions, n_samples, ploidy), np.int32)
    for r in nb.prange(n_regions):
        v_s, v_e = offsets[r], offsets[r + 1]
        n_variants = v_e - v_s
        if n_variants == 0:
            continue
        ref_start = starts[r]
        length = lengths[r]
        for s in nb.prange(n_samples):
            for p in range(ploidy):
                cum_ilen = 0
                rel_ref_end = 0
                cum_write_len = 0
                for rel_v_idx in range(v_s, v_e):
                    if genos[s, p, rel_v_idx] == 1:
                        abs_v_idx = first_v_idxs[r] + rel_v_idx - v_s
                        rel_pos = positions[abs_v_idx] - ref_start
                        maybe_add_one = (
                            rel_pos >= 0
                        )  # length of alt allele is +1 only if rel_pos >= 0
                        ilen: int = ilens[abs_v_idx]  # type: ignore

                        # add dist from last variant to current variant
                        cum_write_len += rel_pos - rel_ref_end

                        # do we need this variant to reach the length?
                        if cum_write_len < length:
                            keep[s, p, rel_v_idx] = True

                            # update rel_ref_end to end of variant
                            v_rel_end = rel_pos - min(0, ilen) + maybe_add_one
                            rel_ref_end = v_rel_end + maybe_add_one

                            # update cum_write_len and cum_ilen
                            missing_len = length - cum_write_len
                            v_len = (
                                max(rel_pos, 0)
                                - rel_ref_end
                                + max(0, ilen)
                                + maybe_add_one
                            )
                            clip_right = max(0, v_len - missing_len)
                            v_len -= clip_right
                            cum_write_len += v_len
                            cum_ilen += ilen - clip_right
                    else:
                        keep[s, p, rel_v_idx] = False
                out_ilens[r, s, p] = cum_ilen

    for r in nb.prange(n_regions):
        for s in nb.prange(n_samples):
            out_ilens[r, s, 0] = out_ilens[r, s, :].min()
        out_ilens[r, 0, 0] = out_ilens[r, :, 0].min()
    min_ilens = out_ilens[:, 0, 0]
    return keep, min_ilens


@nb.njit(parallel=True, nogil=True, cache=True)
def get_n_per_rsp(keep: NDArray[np.bool_], offsets: NDArray[np.int64], n_regions: int):
    n_samples, ploidy, _ = keep.shape
    n_per_rsp = np.empty((n_regions, n_samples, ploidy), np.int32)
    for r in nb.prange(n_regions):
        o_s, o_e = offsets[r], offsets[r + 1]
        for s in nb.prange(n_samples):
            for p in nb.prange(ploidy):
                n_per_rsp[r, s, p] = keep[s, p, o_s:o_e].sum()
    return n_per_rsp


@nb.njit(parallel=True, nogil=True, cache=True)
def keep_mask_to_rsp_v_idx(
    keep: NDArray[np.bool_],  # (s p v)
    first_v_idxs: NDArray[np.int32],  # (r)
    offsets: NDArray[np.int64],  # (r + 1)
    sparse_offsets: NDArray[np.int64],  # (r*s*p + 1)
    n_regions,
    n_samples,
    ploidy,
):
    variant_idxs = np.empty(sparse_offsets[-1], np.int32)
    for r in nb.prange(n_regions):
        fvi = first_v_idxs[r]
        o_s, o_e = offsets[r], offsets[r + 1]
        n_variants = o_e - o_s
        if n_variants == 0:
            continue
        for s in nb.prange(n_samples):
            for p in nb.prange(ploidy):
                out_start = sparse_offsets[r * n_samples * ploidy + s * ploidy + p]
                out_step = 0
                for v in range(n_variants):
                    if keep[s, p, o_s + v]:
                        variant_idxs[out_start + out_step] = fvi + v
                        out_step += 1
    return variant_idxs


@define
class SparseSomaticGenotypes:
    """Sparse, unphased, somatic genotypes corresponding to distinct regions. In this format, genotypes are stored as a ragged 2D array where each
    sample and region may have a different number of variants, since unknown and REF genotypes are not stored. The
    variant indices are aligned to the genotypes. Physically, the genotypes and variant indices are stored as 1D arrays.
    Then, each sample and region's info can be sliced out using the offsets:
    >>> i = np.ravel_multi_index((r, s), (n_regions, n_samples))
    >>> variant_idxs[offsets[i]:offsets[i+1]]
    >>> ccfs[offsets[i]:offsets[i+1]]

    Attributes
    ----------
    variant_idxs : NDArray[np.int32]
        Shape = (variants * samples) Variant indices.
    ccfs : NDArray[np.float32]
        Shape = (variants * samples) Cancer cell fractions (CCF).
    offsets : NDArray[np.int32]
        Shape = (regions * samples * ploidy + 1) Offsets into genos.
    n_samples : int
        Number of samples.
    n_regions : int
        Number of regions.
    """

    variant_idxs: NDArray[np.int32]  # (variants * samples)
    ccfs: NDArray[np.float32]  # (variants * samples)
    offsets: NDArray[np.int64]  # (regions * samples + 1)
    n_regions: int
    n_samples: int
    ploidy = 1

    @property
    def effective_shape(self):
        """Effective shape of the sparse genotypes (n_regions, n_samples, ploidy)
        where ploidy is always represented as 1. The ploidy is treated as 1 to be consistent with."""
        return (self.n_regions, self.n_samples, self.ploidy)

    @classmethod
    def empty(cls, n_regions: int, n_samples: int):
        """Create an empty sparse genotypes object."""
        return cls(
            np.empty(0, np.int32),
            np.empty(0, np.float32),
            np.zeros(n_regions * n_samples + 1, np.int64),
            n_regions,
            n_samples,
        )

    @property
    def is_empty(self) -> bool:
        return len(self.variant_idxs) == 0

    def vars(self, region: int, sample: int):
        """Get variant indices for a given sample and region."""
        i = np.ravel_multi_index((region, sample), (self.n_regions, self.n_samples))
        vars = self.variant_idxs[self.offsets[i] : self.offsets[i + 1]]
        return vars

    def concat(*genos: "SparseSomaticGenotypes") -> "SparseSomaticGenotypes":
        """Concatenate sparse genotypes."""

        if not all(g.n_samples == genos[0].n_samples for g in genos):
            raise ValueError("All genotypes must have the same number of samples.")

        total_n_regions = sum(g.n_regions for g in genos)
        variant_idxs = np.concatenate([g.variant_idxs for g in genos])
        offsets = _lengths_to_offsets(
            np.concatenate([np.diff(g.offsets) for g in genos])
        )

        ccfs = np.concatenate([g.ccfs for g in genos if g.ccfs is not None])

        return SparseSomaticGenotypes(
            variant_idxs=variant_idxs,
            offsets=offsets,
            n_regions=total_n_regions,
            n_samples=genos[0].n_samples,
            ccfs=ccfs,
        )

    @classmethod
    def from_dense(
        cls,
        genos: NDArray[np.int8],
        first_v_idxs: NDArray[np.int32],
        offsets: NDArray[np.int64],
        ccfs: NDArray[np.float32],
    ):
        """Convert dense genotypes to sparse genotypes. Genotypes will be considered ALT if any allele is ALT.
        e.g. 0/1 -> 1, 1/1 -> 1, 0/0 -> 0, 0/2 -> 1.

        Parameters
        ----------
        genos : NDArray[np.int8]
            Shape = (sample, ploidy, variants) Genotypes.
        first_v_idxs : NDArray[np.uint32]
            Shape = (regions) First variant index for each region.
        offsets : NDArray[np.uint32]
            Shape = (regions + 1) Offsets into genos.
        ccfs : Optional[NDArray[np.float32]]
            Shape = (sample, variants) Cancer cell fractions (CCF).
        """
        n_regions = len(first_v_idxs)
        n_samples = genos.shape[0]
        # (s p v) -> (s v)
        keep = (genos == 1).any(1)
        # (r s)
        n_per_rs = get_n_per_rs(keep, offsets, n_regions)
        sparse_offsets = _lengths_to_offsets(n_per_rs.ravel(), np.int64)
        variant_idxs = keep_mask_to_rs_v_idx(
            keep, first_v_idxs, offsets, sparse_offsets, n_regions, n_samples
        )
        # (s v) -> region/variant-major, flattened
        ccfs = ccfs.T[keep.T]
        return cls(
            variant_idxs=variant_idxs,
            ccfs=ccfs,
            offsets=sparse_offsets,
            n_regions=n_regions,
            n_samples=n_samples,
        )

    @classmethod
    def from_dense_with_length(
        cls,
        genos: NDArray[np.int8],
        first_v_idxs: NDArray[np.int32],
        offsets: NDArray[np.int64],
        ilens: NDArray[np.int32],
        positions: NDArray[np.int32],
        starts: NDArray[np.int32],
        lengths: NDArray[np.int32],
        ccfs: NDArray[np.float32],
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
            Shape = (total_variants) ILEN of all unique variants.
        positions : NDArray[np.int32]
            Shape = (total_variants) Positions of unique variants.
        starts : NDArray[np.int32]
            Shape = (regions) Start of query regions.
        lengths : NDArray[np.int32]
            Shape = (regions) Desired lengths of the output haplotypes.
        ccfs : NDArray[np.float32]
            Shape = (sample, variants) Cancer cell fractions (CCF).
        """
        n_regions = len(first_v_idxs)
        n_samples = genos.shape[0]
        # (s p v)
        keep, min_ilens = get_keep_mask_for_length(
            genos,
            offsets,
            first_v_idxs,
            positions,
            ilens,
            starts,
            lengths,
        )
        keep = keep.any(1)
        # (r)
        max_ends: NDArray[np.int32] = starts + lengths - min_ilens.clip(max=0)
        # (r s)
        n_per_rs = get_n_per_rs(keep, offsets, n_regions)
        sparse_offsets = _lengths_to_offsets(n_per_rs.ravel(), np.int64)
        variant_idxs = keep_mask_to_rs_v_idx(
            keep, first_v_idxs, offsets, sparse_offsets, n_regions, n_samples
        )
        # (s v) -> region/variant-major, flattened
        ccfs = ccfs.T[keep.T]
        sparse_genos = cls(
            variant_idxs=variant_idxs,
            ccfs=ccfs,
            offsets=sparse_offsets,
            n_regions=n_regions,
            n_samples=n_samples,
        )
        return sparse_genos, max_ends


@nb.njit(parallel=True, nogil=True, cache=True)
def get_n_per_rs(keep, offsets, n_regions):
    n_samples, n_variants = keep.shape
    n_per_rs = np.empty((n_regions, n_samples), np.int32)
    for r in nb.prange(n_regions):
        o_s, o_e = offsets[r], offsets[r + 1]
        for s in nb.prange(n_samples):
            n_per_rs[r, s] = keep[s, o_s:o_e].sum()
    return n_per_rs


@nb.njit(parallel=True, nogil=True, cache=True)
def keep_mask_to_rs_v_idx(
    keep,  # (s v)
    first_v_idxs,  # (r)
    offsets,  # (r + 1)
    sparse_offsets,  # (r*s + 1)
    n_regions,
    n_samples,
):
    variant_idxs = np.empty(sparse_offsets[-1], np.int32)
    for r in nb.prange(n_regions):
        fvi = first_v_idxs[r]
        o_s, o_e = offsets[r], offsets[r + 1]
        n_variants = o_e - o_s
        if n_variants == 0:
            continue
        for s in nb.prange(n_samples):
            out_start = sparse_offsets[r * n_samples + s]
            out_step = 0
            for v in range(n_variants):
                if keep[s, o_s + v]:
                    variant_idxs[out_start + out_step] = fvi + v
                    out_step += 1
    return variant_idxs


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
    geno_offset_idxs: NDArray[np.intp],
    geno_v_idxs: NDArray[np.int32],
    geno_offsets: NDArray[np.int64],
    size_diffs: NDArray[np.int32],
    keep: Optional[NDArray[np.bool_]] = None,
    keep_offsets: Optional[NDArray[np.int64]] = None,
    starts: Optional[NDArray[np.int32]] = None,
    ends: Optional[NDArray[np.int32]] = None,
    positions: Optional[NDArray[np.int32]] = None,
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
    size_diffs : NDArray[np.int32]
        Shape = (total_variants) Size of all unique variants.
    keep : Optional[NDArray[np.bool_]]
        Shape = (variants*samples*ploidy) Keep mask for genotypes.
    keep_offsets : Optional[NDArray[np.int64]]
        Shape = (regions*samples*ploidy + 1) Offsets into keep.
    starts : Optional[NDArray[np.int32]]
        Shape = (regions) Start of query regions.
    ends : Optional[NDArray[np.int32]]
        Shape = (regions) End of query regions.
    positions : Optional[NDArray[np.int32]]
        Shape = (total_variants) Positions of unique variants.
    """
    n_queries, ploidy = geno_offset_idxs.shape
    diffs = np.empty((n_queries, ploidy), np.int32)
    for query in nb.prange(n_queries):
        for hap in nb.prange(ploidy):
            o_idx = geno_offset_idxs[query, hap]
            o_s, o_e = geno_offsets[o_idx], geno_offsets[o_idx + 1]
            n_variants = o_e - o_s
            if n_variants == 0:
                diffs[query, hap] = 0
            elif starts is not None and ends is not None and positions is not None:
                diffs[query, hap] = 0
                for v in range(o_s, o_e):
                    if keep is not None and keep_offsets is not None:
                        k_s = keep_offsets[query * ploidy + hap]
                        v_keep = keep[k_s + (v - o_s)]
                        if not v_keep:
                            continue

                    v_idx: int = geno_v_idxs[v]
                    v_pos = positions[v_idx]
                    v_diff = size_diffs[v_idx]
                    # +1 assumes atomized variants
                    v_end = v_pos - min(0, v_diff) + 1

                    if v_end <= starts[query]:
                        # variant doesn't span region
                        continue

                    if v_pos >= ends[query]:
                        # variants are sorted by position so this variant and everything
                        # after will be outside the region
                        break

                    # deletion may start before region
                    #     0 1 2 3 4 5 6
                    # DEL s - - r e - - : +max(0, 3 - 0) -> -3 + 3 = 0
                    # DEL r - s - e - - : +max(0, 0 - 2) -> -1 + 0 = -1
                    # where r is region start, s is variant start, e is variant end (exclusive)
                    # count the "-" to get ilen
                    if v_diff < 0:
                        v_diff += max(0, starts[query] - v_pos)
                    # deletion may end after region
                    v_diff += max(0, v_end - ends[query])

                    diffs[query, hap] += v_diff
            elif keep is not None and keep_offsets is not None:
                v_idxs = geno_v_idxs[o_s:o_e]
                k_idx = query * ploidy + hap
                qh_keep = keep[keep_offsets[k_idx] : keep_offsets[k_idx + 1]]
                v_idxs = v_idxs[qh_keep]
                diffs[query, hap] = size_diffs[v_idxs].sum()
            else:
                diffs[query, hap] = size_diffs[geno_v_idxs[o_s:o_e]].sum()
    return diffs


@nb.njit(parallel=True, nogil=True, cache=True)
def reconstruct_haplotypes_from_sparse(
    out: NDArray[np.uint8],
    out_offsets: NDArray[np.int64],
    regions: NDArray[np.int32],
    shifts: NDArray[np.int32],
    geno_offset_idxs: NDArray[np.intp],
    geno_offsets: NDArray[np.int64],
    geno_v_idxs: NDArray[np.int32],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    alt_alleles: NDArray[np.uint8],
    alt_offsets: NDArray[np.int64],
    ref: NDArray[np.uint8],
    ref_offsets: NDArray[np.uint64],
    pad_char: int,
    keep: Optional[NDArray[np.bool_]] = None,
    keep_offsets: Optional[NDArray[np.int64]] = None,
    annot_v_idxs: Optional[NDArray[np.int32]] = None,
    annot_ref_pos: Optional[NDArray[np.int32]] = None,
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
                positions=positions,
                sizes=sizes,
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
    geno_v_idxs: NDArray[np.int32],
    geno_offsets: NDArray[np.int64],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    shift: int,
    alt_alleles: NDArray[np.uint8],  # full set
    alt_offsets: NDArray[np.int64],  # full set
    ref: NDArray[np.uint8],  # full contig
    ref_start: int,  # may be negative
    out: NDArray[np.uint8],
    pad_char: int,
    keep: Optional[NDArray[np.bool_]] = None,
    annot_v_idxs: Optional[NDArray[np.int32]] = None,
    annot_ref_pos: Optional[NDArray[np.int32]] = None,
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
    _variant_idxs = geno_v_idxs[geno_offsets[offset_idx] : geno_offsets[offset_idx + 1]]
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
        v_pos = positions[variant]
        v_diff = sizes[variant]
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

    # fill rest with reference sequence and pad with Ns
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

        # pad
        if out_end_idx < length:
            out[out_end_idx:] = pad_char
            if annot_v_idxs is not None:
                annot_v_idxs[out_end_idx:] = -1
            if annot_ref_pos is not None:
                annot_ref_pos[out_end_idx:] = -1


UNSEEN_VARIANT = np.iinfo(np.uint32).max


@nb.njit(parallel=True, nogil=True, cache=True)
def choose_unphased_variants(
    starts: NDArray[np.int32],
    ends: NDArray[np.int32],
    geno_offset_idxs: NDArray[np.intp],
    geno_v_idxs: NDArray[np.int32],
    geno_offsets: NDArray[np.int64],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    ccfs: NDArray[np.float32],
    deterministic: bool,
) -> Tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """Mark variants to keep for each haplotype.

    Parameters
    ----------
    geno_offset_idxs : NDArray[np.intp]
        Shape = (n_regions, ploidy) Indices for each region into offsets.
    starts : NDArray[np.int32]
        Shape = (n_regions) Start positions for each region.
    offsets : NDArray[np.int64]
        Shape = (total_variants + 1) Offsets into sparse genotypes.
    sparse_genos : NDArray[np.int32]
        Shape = (total_variants) Sparse genotypes i.e. variant indices for ALT genotypes.
    positions : NDArray[np.int32]
        Shape = (total_variants) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (total_variants) Sizes of variants.
    ccfs : NDArray[np.float32]
        Shape = (total_variants) Cancer cell fractions (CCF) of variants.
    ends : NDArray[np.int32]
        Shape = (n_regions) Ends for each region.
    deterministic : bool
        Whether to deterministically assign variants to groups
    """
    n_regions, ploidy = geno_offset_idxs.shape

    lengths = np.empty((n_regions, ploidy), np.int64)
    for query in nb.prange(n_regions):
        for hap in range(ploidy):
            o_idx = geno_offset_idxs[query, hap]
            o_s, o_e = geno_offsets[o_idx], geno_offsets[o_idx + 1]
            lengths[query, hap] = o_e - o_s
    keep_offsets = np.empty(n_regions * ploidy + 1, np.int64)
    keep_offsets[0] = 0
    keep_offsets[1:] = lengths.cumsum()

    n_variants = keep_offsets[-1]
    groups = np.empty(n_variants, np.uint32)
    group_ends = np.empty(n_variants, np.uint32)
    write_lens = np.empty(n_variants, np.uint32)
    keep = np.empty(n_variants, np.bool_)

    for query in nb.prange(n_regions):
        ref_start: int = starts[query]
        ref_end: int = ends[query]
        for hap in nb.prange(ploidy):
            o_idx = geno_offset_idxs[query, hap]
            o_s, o_e = geno_offsets[o_idx], geno_offsets[o_idx + 1]
            qh_genos = geno_v_idxs[o_s:o_e]
            qh_ccfs = ccfs[o_s:o_e]

            k_idx = query * ploidy + hap
            k_s, k_e = keep_offsets[k_idx], keep_offsets[k_idx + 1]
            qh_groups = groups[k_s:k_e]
            qh_ends = group_ends[k_s:k_e]
            qh_w_lens = write_lens[k_s:k_e]
            qh_keep = keep[k_s:k_e]

            qh_keep[:] = _choose_unphased_variants(
                query_start=ref_start,
                query_end=ref_end,
                variant_idxs=qh_genos,
                ccfs=qh_ccfs,
                positions=positions,
                sizes=sizes,
                groups=qh_groups,
                ref_ends=qh_ends,
                write_lens=qh_w_lens,
                deterministic=deterministic,
            )
    return keep, keep_offsets


@nb.njit(nogil=True, cache=True)
def _choose_unphased_variants(
    query_start: int,
    query_end: int,
    variant_idxs: NDArray[np.int32],  # (v)
    ccfs: NDArray[np.float32],  # (v)
    positions: NDArray[np.int32],  # (total variants)
    sizes: NDArray[np.int32],  # (total variants)
    groups: NDArray[np.uint32],  # (v)
    ref_ends: NDArray[np.uint32],  # (g)
    write_lens: NDArray[np.uint32],  # (g)
    deterministic: bool,
) -> NDArray[np.bool_]:
    # no variants
    if len(variant_idxs) == 0:
        return np.ones(0, np.bool_)

    # treat missing CCF as 1.0
    ccfs = np.nan_to_num(ccfs, True, 1.0)
    groups[:] = UNSEEN_VARIANT
    ref_ends[:] = query_start
    write_lens[:] = 0
    n_groups = 0

    # Assign variants to groups
    # stop once the total written length for all groups is >= length
    for v in range(len(variant_idxs)):
        n_compat = 0
        v_idx: int = variant_idxs[v]
        v_pos = positions[v_idx]
        # +1 assumes atomized variants
        maybe_add_one = int(v_pos >= query_start)
        v_ref_end = v_pos - min(0, sizes[v_idx]) + 1

        if v_ref_end <= query_start:
            # skip the variant by leaving its group as unseen
            continue

        if v_pos >= query_end:
            # variants are sorted by position, everything after this will be outside the query
            break

        # choose group for variant
        for g in range(n_groups):
            # variant compatible with group
            if v_pos >= ref_ends[g]:
                n_compat += 1

                # unseen variant, assign it to first compatible group
                if groups[v] == UNSEEN_VARIANT:
                    groups[v] = g
                # otherwise randomly assign variant to compatible group
                # 1/n_compat chance ensures uniform choice across groups
                elif not deterministic and random.random() < 1 / n_compat:
                    groups[v] = g

        # variant not compatible with any group or there are no groups, make new group
        if groups[v] == UNSEEN_VARIANT:
            n_groups += 1
            groups[v] = n_groups - 1

        # finished with variant assignment

        # update group info
        # writable length = length of variant + dist from last v_ref_end or the missing length, whichever is smaller
        v_group = groups[v]
        # max(v_pos, ref_start) addresses a spanning deletion from before ref_start
        # likewise, (v_pos > ref_start) accounts for a spanning deletion that starts after ref_start
        # spanning deletions will have a write_len of 0 unlike all other variants where we always write at least 1 nt
        # Note that this also assumes no MNPs, only SNPs and INDELs. Can relax this by passing in the alleles similar
        # to above.
        v_write_len = (
            max(v_pos, query_start)
            - ref_ends[v_group]
            + max(0, sizes[v_idx])
            + maybe_add_one
        )
        writable_len = v_write_len
        write_lens[v_group] += writable_len
        ref_ends[v_group] = v_ref_end

    # If not all groups have write_len = target_len after seeing all variants,
    # that's ok since this won't affect the group selection process in the next step.

    if n_groups == 1:
        return groups == 0

    # Choose a group proportional to total ccf normalized by reference length.
    # This is because variants with long ref len will prevent other variants from
    # being assigned to the same group, reducing the potential total ccf of the
    # group.f
    cum_prop = write_lens[:n_groups].view(
        np.float32
    )  # reinterpret this memory to avoid allocation
    for g in range(n_groups):
        v_starts = positions[variant_idxs[groups == g]]
        v_ends = (
            v_starts
            - np.minimum(0, sizes[variant_idxs[groups == g]])
            + (v_starts >= query_start)
        )
        ref_lengths = np.minimum(v_ends, ref_ends[g]) - np.maximum(
            v_starts, query_start
        )
        cum_prop[g] = (ccfs[groups == g] / ref_lengths).sum()
    if deterministic:
        keep_group = cum_prop.argmax()
    else:
        cum_prop = cum_prop.cumsum()
        cum_prop /= cum_prop[-1]
        keep_group = (random.random() <= cum_prop).sum() - 1

    return groups == keep_group
