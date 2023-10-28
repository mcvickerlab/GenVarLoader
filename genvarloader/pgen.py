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

        pvar_path = path.with_suffix(".pvar")
        pvar_arrow_path = path.with_suffix(".gvl.arrow")
        ends_arrow_path = path.with_suffix(".ends.gvl.arrow")
        # exists and was modified more recently than .pvar
        if (
            pvar_arrow_path.exists()
            and pvar_arrow_path.stat().st_mtime > pvar_path.stat().st_mtime
            and ends_arrow_path.exists()
            and ends_arrow_path.stat().st_mtime > pvar_path.stat().st_mtime
        ):
            pvar = pl.read_ipc(pvar_arrow_path)
            ends = pl.read_ipc(ends_arrow_path)
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
            )

            if (pvar["ALT"].str.contains(",")).any():
                raise RuntimeError(
                    f"""PGEN file {path} contains multi-allelic variants which are not 
                    yet supported by GenVarLoader. Split your multi-allelic variants 
                    with `bcftools norm -a --atom-overlaps . -m - <file.vcf>` then 
                    remake the PGEN file with the `--vcf-half-call r` option."""
                )

            pvar = pvar.with_columns(
                POS=pl.col("POS") - 1,
                ILEN=(
                    pl.col("ALT").str.len_bytes().cast(pl.Int32)
                    - pl.col("REF").str.len_bytes().cast(pl.Int32)
                ),
            )

            # ends in reference coordiantes, 0-based **inclusive**
            # end_to_var_idx is a mapping from variants sorted by end to the earliest
            # positioned variant that has an end >= the end of the variant at that index
            # e.g. if the v0 has end 200, and v1 has end 100, then ends would be sorted
            # as [v1, v0] and end_to_var_idx[0] = 1 and end_to_var_idx[1] = 1
            ends = (
                pvar.with_row_count("VAR_IDX")
                .with_columns(
                    END=pl.col("POS") - pl.col("ILEN").clip_max(0)  #! end-inclusive
                )
                # variants are sorted by pos
                .filter(pl.col("POS") == pl.col("POS").min().over("#CHROM", "END"))
                .select("#CHROM", "END", "VAR_IDX")
            )
            ends = (
                ends.sort("#CHROM", "END")
                .group_by("#CHROM", maintain_order=True)
                .agg(
                    pl.all().sort_by("END"),
                )
                .explode(pl.exclude("#CHROM"))
                .group_by("#CHROM", maintain_order=True)
                .agg(
                    pl.all(),
                    pl.col("VAR_IDX")
                    .reverse()
                    .rolling_min(ends.height, min_periods=1)
                    .reverse()
                    .alias("END_TO_VAR_IDX"),
                )
                .explode(pl.exclude("#CHROM"))
                .select("END", "END_TO_VAR_IDX")
            )
            pvar.write_ipc(pvar_arrow_path)
            ends.write_ipc(ends_arrow_path)

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
        self.ends = ends["END"].to_numpy()
        self.end_to_var_idx = ends["END_TO_VAR_IDX"].to_numpy()
        self.end_contig_offsets = np.zeros(len(self.contig_idx) + 1, dtype=np.uint32)
        self.end_contig_offsets[1:] = (
            pvar["#CHROM"].set_sorted().unique_counts().cumsum().to_numpy()
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

        out: List[Optional[DenseGenotypes]] = [None] * len(starts)

        # get variant positions and indices
        c_idx = self.contig_idx.get(contig, None)

        # contig is not present in PGEN, has no variants
        if c_idx is None:
            return out

        c_slice = slice(self.contig_offsets[c_idx], self.contig_offsets[c_idx + 1])
        end_c_slice = slice(
            self.end_contig_offsets[c_idx], self.end_contig_offsets[c_idx + 1]
        )

        # add c_slice.start to account for the contig offset
        _s_idxs = np.searchsorted(self.ends[end_c_slice], starts) + end_c_slice.start
        s_idxs = self.end_to_var_idx[_s_idxs]
        e_idxs = np.searchsorted(self.positions[c_slice], ends) + c_slice.start

        min_s_idx = s_idxs.min()
        max_e_idx = e_idxs.max()

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

        ploid = kwargs.get("ploid", None)
        if ploid is None:
            ploid = np.arange(self.ploidy)
        else:
            ploid = np.asarray(ploid)

        starts, ends = np.asarray(starts, dtype=np.int64), np.asarray(
            ends, dtype=np.int64
        )

        contig = self.normalize_contig_name(contig)

        out: List[Optional[DenseGenotypes]] = [None] * len(starts)

        # get variant positions and indices
        c_idx = self.contig_idx.get(contig, None)

        # contig is not present in PGEN, has no variants
        if c_idx is None:
            return out, ends

        c_slice = slice(self.contig_offsets[c_idx], self.contig_offsets[c_idx + 1])

        # no variants in contig
        if c_slice.start == c_slice.stop:
            return out, ends

        end_c_slice = slice(
            self.end_contig_offsets[c_idx], self.end_contig_offsets[c_idx + 1]
        )

        effective_starts = ends - target_length
        # idxs are relative to unique contig ends
        _eff_s_idxs = (
            np.searchsorted(self.ends[end_c_slice], effective_starts)
            + end_c_slice.start
        )
        # make idxs relative to contig variants
        eff_s_idxs = self.end_to_var_idx[_eff_s_idxs] - c_slice.start

        max_ends, e_idxs = get_ends_and_idxs(
            effective_starts,
            eff_s_idxs,
            self.positions[c_slice],
            self.size_diffs[c_slice],
            target_length,
        )
        # make idxs absolute
        e_idxs += c_slice.start

        # idxs are relative to unique ends
        _s_idxs = np.searchsorted(self.ends[end_c_slice], starts) + end_c_slice.start
        # turn into variant indices, self.ends is sorted by chrom->end, whereas variants
        # are sorted by chrom->start
        s_idxs = self.end_to_var_idx[_s_idxs]

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
                    genotypes=genotypes[:, ploid, rel_s_idx:rel_e_idx],
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
        var_pos = positions[0]
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
            var_pos = positions[var_idx]
            var_diff = size_diffs[var_idx]
            # pos + 1 nt of ref - var_diff (neg for dels)
            var_end = var_pos + 1 - var_diff

            if var_pos < start:
                # could get a del_end <= start because variants are sorted by
                # pos->ends. even tho start_idx is the first end to span start,
                # variants that do not span start can come after it
                var_diff = min(0, start - var_end)

            if var_diff < 0:
                # split multiallelic, use the largest del at the site
                # they are sorted so the last one is largest
                if var_pos == prev_del_pos:
                    prev_del_diff = var_diff
                    prev_del_pos = var_pos
                    prev_del_end = var_end
                # overlapping deletion
                elif var_pos < prev_del_end:
                    # ignore del if it's larger (i.e. shorter)
                    if var_diff >= prev_del_diff:
                        continue
                    # use this deletion instead
                    else:
                        prev_del_diff = var_diff
                        prev_del_pos = var_pos
                        prev_del_end = var_end
                # prev_del is the largest del from the prev position
                elif var_pos >= prev_del_end:
                    # increment total_del by var_diff (negative for dels)
                    total_del -= prev_del_diff
                    # update prev_del
                    prev_del_diff = var_diff
                    prev_del_pos = var_pos
                    prev_del_end = var_end

                length = var_pos - start - total_del
                if length >= target_length:
                    break
            else:
                length = var_pos - start - total_del
                if length > target_length:
                    # stop at this variant
                    var_idx -= 1
                    break

        total_del -= prev_del_diff
        ref_pos = prev_del_end
        length = ref_pos - start - total_del
        ref_pos += target_length - length

        max_ends[i] = ref_pos
        end_idxs[i] = var_idx + 1

    return max_ends, end_idxs
