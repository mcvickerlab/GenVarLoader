from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

        Notes
        -----
        Writes a copy of the .pvar file as an Arrow file (.gvl.arrow) to speed up
        loading (by 25x or more for larger files.)
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
            if len(_samples) != len(samples):
                raise ValueError("Got samples that are not in the pgen file.")
            self.samples = _samples
            self.sample_idx = sample_idx.astype(np.uint32)
        else:
            self.samples = psam_samples
            self.sample_idx = np.arange(len(psam_samples), dtype=np.uint32)
        self.n_samples = len(self.samples)

        pvar_arrow_path = path.with_suffix(".gvl.arrow")
        pvar_path = path.with_suffix(".pvar")
        # exists and was modified more recently than .pvar
        if (
            pvar_arrow_path.exists()
            and pvar_arrow_path.stat().st_mtime > pvar_path.stat().st_mtime
        ):
            pvar = pl.read_ipc(pvar_arrow_path)
        else:
            with open(pvar_path, "r") as f:
                skip_rows = 0
                while f.readline().startswith("##"):
                    skip_rows += 1

            pvar = pl.read_csv(
                pvar_path,
                separator="\t",
                skip_rows=skip_rows,
                columns=["#CHROM", "POS", "REF", "ALT"],
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
                    f"""PGEN file {path} contains multi-allelic variants which are not 
                    yet supported by GenVarLoader. Split your multi-allelic variants 
                    with `bcftools norm -a --atom-overlaps . -m - <file.vcf>` then 
                    remake the PGEN file with the `--vcf-half-call r` option."""
                )

            pvar.write_ipc(pvar_arrow_path)

        contigs = pvar["#CHROM"].set_sorted()
        self.contig_idx: Dict[str, int] = {
            c: i for i, c in enumerate(contigs.unique(maintain_order=True))
        }
        # (c+1)
        self.contig_offsets = np.zeros(len(self.contig_idx) + 1, dtype=np.uint32)
        self.contig_offsets[1:] = contigs.unique_counts().cumsum().to_numpy()
        # (v)
        self.positions: NDArray[np.int32] = pvar["POS"].to_numpy()
        # no multi-allelics
        self.ref = VLenAlleles.from_polars(pvar["REF"])
        self.alt = VLenAlleles.from_polars(pvar["ALT"])
        self.size_diffs: NDArray[np.int32] = pvar["ILEN"].to_numpy()

        # ends in reference coordiantes, 0-based exclusive
        for_ends = (
            pvar.with_row_count("var_nr")
            .with_columns(
                END=pl.col("POS") - pl.col("ILEN").clip_max(0) + 1  # 1 nt of ref
            )
            # variants are sorted by pos
            .filter(pl.col("POS") == pl.col("POS").min().over("#CHROM", "END"))
            .group_by("#CHROM", maintain_order=True)
            .agg(pl.all().sort_by("END"))
            .explode(pl.exclude("#CHROM"))
        )
        self.ends = for_ends["END"].to_numpy()
        self.end_nr = np.empty(len(self.ends) + 1, dtype=np.uint32)
        self.end_nr[-1] = pvar.height
        self.end_nr[:-1] = for_ends["var_nr"].cast(pl.UInt32).to_numpy()
        self.end_contig_offsets = np.zeros(len(self.contig_idx) + 1, dtype=np.uint32)
        self.end_contig_offsets[1:] = (
            for_ends["#CHROM"].set_sorted().unique_counts().cumsum().to_numpy()
        )

        self.contig_starts_with_chr = self.infer_contig_prefix(self.contig_idx.keys())

    def _pgen(self, sample_idx: Optional[NDArray[np.uint32]]):
        if sample_idx is not None:
            sample_idx = np.sort(sample_idx)
        return pgenlib.PgenReader(bytes(self.pgen_path), sample_subset=sample_idx)

    def read(
        self, contig: str, starts: NDArray[np.int64], ends: NDArray[np.int64], **kwargs
    ) -> List[Optional[DenseGenotypes]]:
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
        end_c_slice = slice(
            self.end_contig_offsets[c_idx], self.end_contig_offsets[c_idx + 1]
        )

        # add c_slice.start to account for the contig offset
        _s_idxs = (
            np.searchsorted(self.ends[end_c_slice], starts, side="right")
            + end_c_slice.start
        )
        s_idxs = self.end_nr[_s_idxs]
        e_idxs = np.searchsorted(self.positions[c_slice], ends) + c_slice.start

        min_s_idx = s_idxs.min()
        max_e_idx = e_idxs.max()

        out: List[Optional[DenseGenotypes]] = [None] * len(starts)

        if min_s_idx == max_e_idx:
            return out

        # get alleles
        genotypes = np.empty(
            (max_e_idx - min_s_idx, n_samples * self.ploidy), dtype=np.int32
        )
        with self._pgen(pgen_idx) as f:
            # (v s*2)
            f.read_alleles_range(min_s_idx, max_e_idx, genotypes)

        genotypes = genotypes.astype(np.int8)
        # (s*2 v)
        genotypes = genotypes.swapaxes(0, 1)
        # (s 2 v)
        genotypes = np.stack([genotypes[::2], genotypes[1::2]], axis=1)

        # re-order samples to be in query order
        if query_idx is not None:
            genotypes = genotypes[query_idx]

        for i, (min_s_idx, max_e_idx) in enumerate(zip(s_idxs, e_idxs)):
            rel_s_idx = min_s_idx - min_s_idx
            rel_e_idx = max_e_idx - min_s_idx
            if min_s_idx == max_e_idx:
                out[i] = None
            else:
                out[i] = DenseGenotypes(
                    positions=self.positions[min_s_idx:max_e_idx],
                    size_diffs=self.size_diffs[min_s_idx:max_e_idx],
                    ref=self.ref[min_s_idx:max_e_idx],
                    alt=self.alt[min_s_idx:max_e_idx],
                    genotypes=genotypes[..., rel_s_idx:rel_e_idx],
                )

        return out

    def read_for_haplotype_construction(
        self,
        contig: str,
        starts: NDArray[np.int64],
        ends: NDArray[np.int64],
        target_length: int,
        **kwargs,
    ) -> Tuple[List[Optional[DenseGenotypes]], NDArray[np.int64]]:
        """Read genotypes for haplotype construction. This is a special case of `read` 
        where variants beyond the query regions are included to ensure that haplotypes 
        of `target_length` can be constructed. This is necessary because deletions can
        shorten the haplotype, so variants downstream of `end` may be needed to add more
        sequence to the haplotype.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : NDArray[int32]
            Start coordinates, 0-based.
        ends : int, NDArray[int32]
            End coordinates, 0-based exclusive.
        target_length : int
            Target length of the reconstructed haplotypes.

        Returns
        -------
        List[Optional[DenseGenotypes]]
            Genotypes for each query region.
        NDArray[np.int64]
            New ends for querying the reference genome such that enough sequence is 
            available to get haplotypes of `target_length`.
        """ """"""
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
        end_c_slice = slice(
            self.end_contig_offsets[c_idx], self.end_contig_offsets[c_idx + 1]
        )

        out: List[Optional[DenseGenotypes]] = [None] * len(starts)

        # no variants in contig
        if c_slice.start == c_slice.stop:
            return out, ends

        effective_starts = ends - target_length
        # idxs are relative to unique contig ends
        _eff_s_idxs = (
            np.searchsorted(self.ends[end_c_slice], effective_starts, side="right")
            + end_c_slice.start
        )
        # idxs are relative to contig variants
        eff_s_idxs = self.end_nr[_eff_s_idxs] - c_slice.start

        max_ends, e_idxs = get_ends_and_idxs(
            effective_starts,
            eff_s_idxs,
            self.positions[c_slice],
            self.size_diffs[c_slice],
            target_length,
        )

        # idxs are relative to unique ends
        _s_idxs = (
            np.searchsorted(self.ends[end_c_slice], starts, side="right")
            + end_c_slice.start
        )
        # turn into variant indices, self.ends is sorted by chrom->end, whereas variants
        # are sorted by chrom->start
        s_idxs = self.end_nr[_s_idxs]

        min_s_idx = s_idxs.min()
        max_e_idx = e_idxs.max()

        # no variants in query regions
        if max_e_idx <= min_s_idx:
            return out, ends

        # get alleles
        genotypes = np.empty(
            (max_e_idx - min_s_idx, n_samples * self.ploidy), dtype=np.int32
        )
        with self._pgen(pgen_idx) as f:
            # (v s*2)
            f.read_alleles_range(min_s_idx, max_e_idx, genotypes)

        genotypes = genotypes.astype(np.int8)
        # (s*2 v)
        genotypes = genotypes.swapaxes(0, 1)
        # (s 2 v)
        genotypes = np.stack([genotypes[::2], genotypes[1::2]], axis=1)

        # re-order samples to be in query order
        if query_idx is not None:
            genotypes = genotypes[query_idx]

        for i, (s_idx, e_idx) in enumerate(zip(s_idxs, e_idxs)):
            rel_s_idx = s_idx - min_s_idx
            rel_e_idx = e_idx - min_s_idx
            if s_idx == e_idx:
                out[i] = None
            else:
                out[i] = DenseGenotypes(
                    positions=self.positions[s_idx:e_idx],
                    size_diffs=self.size_diffs[s_idx:e_idx],
                    ref=self.ref[s_idx:e_idx],
                    alt=self.alt[s_idx:e_idx],
                    genotypes=genotypes[..., rel_s_idx:rel_e_idx],
                )

        return (out, max_ends)

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


# note this operates on a contigs relative idxs, positions, and size_diffs
@nb.njit(nogil=True, cache=True)
def get_ends_and_idxs(
    starts: NDArray[np.int64],
    start_idxs: NDArray[np.uint32],
    positions: NDArray[np.int32],
    size_diffs: NDArray[np.int32],
    target_length: int,
) -> Tuple[NDArray[np.int64], NDArray[np.uint32]]:
    max_ends = starts + target_length
    end_idxs = start_idxs.copy()

    if len(positions) == 1:
        del_pos = positions[0]
        var_diff = size_diffs[0]
        max_ends[:] = starts + target_length
        if var_diff < 0:
            max_ends -= var_diff
        end_idxs[:] = start_idxs + 1
        return max_ends, end_idxs

    for i in nb.prange(len(start_idxs)):
        # positive, how much deleted
        total_del = 0
        start = starts[i]
        ref_pos = start
        length = 0
        var_idx = start_idxs[i]

        # no variants
        if var_idx >= len(positions):
            continue

        var_diff = size_diffs[var_idx]
        prev_del_pos = positions[var_idx]
        prev_del_end = prev_del_pos
        prev_del_diff = 0

        # no variants in span of start + target_length
        if prev_del_pos >= start + target_length:
            continue

        for var_idx in range(start_idxs[i], len(positions)):
            var_diff = size_diffs[var_idx]
            del_pos = positions[var_idx]
            # pos + 1 nt of ref - var_diff (neg for dels)
            del_end = del_pos + 1 - var_diff

            if del_pos < start:
                # could get a del_end <= start because variants are sorted by
                # pos->ends. even tho start_idx is the first end to span start,
                # variants that do not span start can come after it
                var_diff = min(0, start - del_end)

            # not a del
            if var_diff >= 0:
                continue

            # split multiallelic, use the largest del at the site
            # they are sorted so the last one is largest
            if del_pos == prev_del_pos:
                prev_del_diff = var_diff
                prev_del_pos = del_pos
                prev_del_end = del_end
            # overlapping deletion
            elif del_pos < prev_del_end:
                # ignore del if it's larger (i.e. shorter)
                if var_diff >= prev_del_diff:
                    continue
                # use this deletion instead
                else:
                    prev_del_diff = var_diff
                    prev_del_pos = del_pos
                    prev_del_end = del_end
            # prev_del is the largest del from the prev position
            elif del_pos >= prev_del_end:
                # increment total_del by var_diff (negative for dels)
                total_del -= prev_del_diff
                # update prev_del
                prev_del_diff = var_diff
                prev_del_pos = del_pos
                prev_del_end = del_end

            ref_pos = prev_del_end
            length = ref_pos - start - total_del
            if length >= target_length:
                break

        # no deletions
        if prev_del_diff >= 0:
            continue

        total_del -= prev_del_diff
        ref_pos = prev_del_end
        length = ref_pos - start - total_del
        ref_pos += target_length - length

        max_ends[i] = ref_pos
        end_idxs[i] = var_idx + 1

    return max_ends, end_idxs


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
