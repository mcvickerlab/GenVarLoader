from pathlib import Path
from typing import Dict, List, Optional, Union

import numba as nb
import numpy as np
import polars as pl
from numpy.typing import NDArray

from .types import DenseGenotypes, Variants, VLenAlleles

try:
    import pgenlib

    PGENLIB_INSTALLED = True
except ImportError:
    PGENLIB_INSTALLED = False


class Pgen(Variants):
    def __init__(
        self, path: Union[str, Path], samples: Optional[List[str]] = None
    ) -> None:
        """Reads genotypes from PGEN files. Currently does not support multi-allelic
        sites, but does support **split** multi-allelic sites. This can be done by
        running `bcftools norm -a --atom-overlaps . -m - <file.vcf>` and creating the
        PGEN with the `--vcf-half-call r` option.

        Parameters
        ----------
        path : Union[str, Path]
            Path to any of the PGEN files (.pgen, .pvar, .psam) or their prefix.
        samples : Optional[List[str]], optional
            Which samples to include, by default all samples.
        """
        if not PGENLIB_INSTALLED:
            raise ImportError("Pgenlib must be installed to read PGEN files.")

        # pgenlib is exclusively diploid
        self.ploidy = 2
        # unknown genotypes are set to -9
        self.UNKNOWN = -9

        path = Path(path)
        self.pgen_path = path.with_suffix(".pgen")

        try:
            psam_samples = pl.read_csv(
                path.with_suffix(".psam"), separator="\t", columns=["IID"]
            )["IID"].to_numpy()
        except pl.ColumnNotFoundError:
            psam_samples = pl.read_csv(
                path.with_suffix(".psam"), separator="\t", columns=["#IID"]
            )["#IID"].to_numpy()
        if samples is not None:
            _samples, sample_idx, _ = np.intersect1d(
                psam_samples, samples, return_indices=True
            )
            if len(_samples) == len(samples):
                raise ValueError("Got samples that are not in the pgen file.")
            self.samples = _samples
            self.sample_idx = sample_idx.astype(np.uint32)
        else:
            self.samples = psam_samples
            self.sample_idx = np.arange(len(psam_samples), dtype=np.uint32)
        self.n_samples = len(self.samples)

        pvar_path = path.with_suffix(".pvar")
        with open(pvar_path, "r") as f:
            skip_rows = 0
            while f.readline().startswith("##"):
                skip_rows += 1

        pvar = pl.read_csv(
            pvar_path,
            separator="\t",
            skip_rows=skip_rows,
            dtypes={"#CHROM": pl.Utf8, "POS": pl.Int32},
        ).with_columns(
            POS=pl.col("POS") - 1,
            ILEN=(
                pl.col("ALT").str.lengths().cast(pl.Int32)
                - pl.col("REF").str.lengths().cast(pl.Int32)
            ),
        )

        if (pvar["ALT"].str.contains(",")).any():
            raise RuntimeError(
                f"""PGEN file {path} contains multi-allelic variants which are not yet 
                supported by GenVarLoader. Split your multi-allelic variants with 
                `bcftools norm -a --atom-overlaps . -m - <file.vcf>` then remake the 
                PGEN file with the `--vcf-half-call r` option."""
            )

        contigs = pvar["#CHROM"].set_sorted()
        self.contig_idx: Dict[str, int] = {
            c: i for i, c in enumerate(contigs.unique(maintain_order=True))
        }
        # (c+1)
        self.contig_offsets = np.zeros(len(self.contig_idx) + 1, dtype=np.uint32)
        self.contig_offsets[1:] = contigs.unique_counts().cumsum().to_numpy()
        # (v)
        self.positions: NDArray[np.int32] = pvar["POS"].to_numpy()
        # (v 2), assumes only 1 REF and 1 ALT, no multi-allelics, only SNPs
        self.ref = VLenAlleles.from_polars(pvar["REF"])
        self.alt = VLenAlleles.from_polars(pvar["ALT"])
        self.sizes: NDArray[np.int32] = pvar["ILEN"].to_numpy()
        self.contig_starts_with_chr = self.infer_contig_prefix(self.contig_idx)

    def _pgen(self, sample_idx: Optional[NDArray[np.uint32]]):
        if sample_idx is not None:
            sample_idx = np.sort(sample_idx)
        return pgenlib.PgenReader(bytes(self.pgen_path), sample_subset=sample_idx)

    def read(
        self, contig: str, start: int, end: int, **kwargs
    ) -> Optional[DenseGenotypes]:
        samples = kwargs.get("sample", None)
        if samples is None:
            n_samples = self.n_samples
            pgen_idx, query_idx = None, None
        else:
            n_samples = len(samples)
            pgen_idx, query_idx = self.get_sample_idx(samples)

        contig = self.normalize_contig_name(contig)

        # get variant positions and indices
        c_idx = self.contig_idx[contig]
        c_slice = slice(self.contig_offsets[c_idx], self.contig_offsets[c_idx + 1])

        s_idx, e_idx = (
            np.searchsorted(self.positions[c_slice], [start, end]) + c_slice.start
        )

        end_of_var_before_start = self.positions[s_idx - 1] - self.sizes[s_idx - 1]
        if s_idx > c_slice.start and end_of_var_before_start > start:
            s_idx -= 1

        if s_idx == e_idx:
            return

        q_sizes = self.sizes[s_idx:e_idx].copy()
        if self.positions[s_idx] < start:
            q_sizes[0] = start - end_of_var_before_start
        max_end = max(
            end, self.positions[e_idx - 1] - self.sizes[e_idx - 1]
        ) - q_sizes.sum(where=q_sizes < 0)

        # get alleles
        with self._pgen(pgen_idx) as f:
            # (v s*2)
            genotypes = np.empty(
                (e_idx - s_idx, n_samples * self.ploidy), dtype=np.int32
            )
            f.read_alleles_range(s_idx, e_idx, genotypes)

        # (s*2 v)
        genotypes = genotypes.swapaxes(0, 1)
        # (s 2 v)
        genotypes = np.stack([genotypes[::2], genotypes[1::2]], 1)
        genotypes = genotypes.astype(np.int8)

        # re-order samples to be in query order
        if query_idx is not None:
            genotypes = genotypes[query_idx]

        return DenseGenotypes(
            self.positions[s_idx:e_idx],
            q_sizes,
            self.ref[s_idx:e_idx],
            self.alt[s_idx:e_idx],
            genotypes,
            max_end,
        )

    def get_sample_idx(self, samples):
        _samples, _pgen_idx, query_idx = np.intersect1d(
            self.samples, samples, return_indices=True
        )
        if len(_samples) != len(samples):
            unrecognized_samples = set(samples) - set(_samples)
            raise ValueError(
                f"""Got samples that are not in the pgen file: 
                {unrecognized_samples}."""
            )
        pgen_idx: NDArray[np.uint32] = self.sample_idx[_pgen_idx]
        return pgen_idx, query_idx


@nb.njit(nogil=True)
def group_duplicates(positions: NDArray[np.int32]):
    """Get the start:end indices of duplicate positions"""
    n = len(positions)

    # Initialize an array to store the groups
    groups = np.empty(n + 1, dtype=np.int32)

    # Initialize variables
    current_group = 0
    current_value = positions[0]

    # Iterate through the array to find groups
    groups[0] = 0
    for i in range(1, n):
        if positions[i] != current_value:
            current_group += 1
            groups[current_group] = i
            current_value = positions[i]

    # Set the end of the last group
    groups[current_group + 1] = n

    # (v+1) i32
    return groups[: current_group + 2]


@nb.njit(nogil=True)
def genos_to_alleles(
    groups: NDArray[np.int32],
    alleles: NDArray[np.uint8],
    genotypes: NDArray[np.int8],
):
    # groups (v+1) i32
    # positions (v) i32
    # alleles (v p) u8
    # genotypes (s p v) i8
    n_variants = len(groups) - 1
    n_samples = len(genotypes)
    ploidy = 2
    UNKNOWN = -9

    out_alleles = np.empty((n_samples, ploidy, n_variants), np.uint8)

    for sample in nb.prange(n_samples):
        for variant in nb.prange(n_variants):
            # groups of duplicate positions are defined by start : end indices
            # provided in dup_groups
            start_idx, end_idx = groups[variant], groups[variant + 1]
            n_dups = end_idx - start_idx

            # (p n_dups)
            variant_geno = genotypes[sample, :, start_idx:end_idx]

            # (p)
            variant_alleles = out_alleles[sample, :, variant]

            # A biallelic variant
            if n_dups == 1:
                biallelic(ploidy, alleles, start_idx, variant_geno, variant_alleles)
                continue

            # Handle half-calls, caused by splitting multiallelic variants
            is_half_call = ~(variant_geno == UNKNOWN).any()
            if is_half_call:
                # merge alt alleles into a single allele, ignoring unknowns
                # assumes each haplotypes has exactly one occurrence of an ALT allele
                half_call(
                    ploidy, n_dups, variant_geno, alleles, start_idx, variant_alleles
                )
                continue

            # A normal variant that happens to overlap with a split multiallelic
            # Catch case where sample has no genotype at the site
            overlap_split_multiallelic(
                n_dups, ploidy, variant_geno, alleles, start_idx, variant_alleles
            )

    return out_alleles


@nb.njit(nogil=True)
def biallelic(ploidy, alleles, start_idx, variant_geno, variant_alleles):
    UNKNOWN = -9
    for hap in nb.prange(ploidy):
        geno = variant_geno[hap, 0]
        if geno == UNKNOWN:
            variant_alleles[hap] = 0
        else:
            variant_alleles[hap] = alleles[start_idx, geno]


@nb.njit(nogil=True)
def half_call(ploidy, n_dups, variant_geno, alleles, start_idx, variant_alleles):
    for hap in nb.prange(ploidy):
        for i in nb.prange(n_dups):
            if variant_geno[hap, i] == 1:
                variant_alleles[hap] = alleles[start_idx + i, 1]


@nb.njit(nogil=True)
def overlap_split_multiallelic(
    n_dups, ploidy, variant_geno, alleles, start_idx, variant_alleles
):
    UNKNOWN = -9
    genotyped_idx = np.int16(-1)
    for i in nb.prange(n_dups):
        n_unknown = np.uint8(0)
        for hap in nb.prange(ploidy):
            if variant_geno[hap, i] != UNKNOWN:
                n_unknown += 1
        if n_unknown == ploidy:
            genotyped_idx = i
    if genotyped_idx == -1:
        for hap in nb.prange(ploidy):
            variant_alleles[hap] = alleles[start_idx, 0]
    else:
        for hap in nb.prange(ploidy):
            variant_alleles[hap] = alleles[
                start_idx + genotyped_idx, variant_geno[hap, genotyped_idx]
            ]
